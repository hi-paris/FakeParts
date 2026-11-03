from typing import List, Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import InstructBlipQFormerConfig, InstructBlipQFormerModel, AutoTokenizer

from utils.utils import (DEFAULT_IM_END_TOKEN, DEFAULT_IM_START_TOKEN,
                         DEFAULT_IMAGE_PATCH_TOKEN)

from .llava.model.language_model.llava_llama import (LlavaLlamaForCausalLM,
                                                     LlavaLlamaModel)

from .segment_anything import build_sam_vit_h

import itertools

def dice_loss(
    inputs: torch.Tensor,
    targets: torch.Tensor,
    num_masks: float,
    scale=1000,  # 100000.0,
    eps=1e-6,
):
    """
    Compute the DICE loss, similar to generalized IOU for masks
    Args:
        inputs: A float tensor of arbitrary shape.
                The predictions for each example.
        targets: A float tensor with the same shape as inputs. Stores the binary
                 classification label for each element in inputs
                (0 for the negative class and 1 for the positive class).
    """
    inputs = inputs.sigmoid()
    inputs = inputs.flatten(1, 2)
    targets = targets.flatten(1, 2)
    numerator = 2 * (inputs / scale * targets).sum(-1)
    denominator = (inputs / scale).sum(-1) + (targets / scale).sum(-1)
    loss = 1 - (numerator + eps) / (denominator + eps)
    loss = loss.sum() / (num_masks + 1e-8)
    return loss


def sigmoid_ce_loss(
    inputs: torch.Tensor,
    targets: torch.Tensor,
    num_masks: float,
):
    """
    Args:
        inputs: A float tensor of arbitrary shape.
                The predictions for each example.
        targets: A float tensor with the same shape as inputs. Stores the binary
                 classification label for each element in inputs
                (0 for the negative class and 1 for the positive class).
    Returns:
        Loss tensor
    """
    loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")
    loss = loss.flatten(1, 2).mean(1).sum() / (num_masks + 1e-8)
    return loss

class SidaMetaModel:
    def __init__(
        self,
        config,
        **kwargs,
    ):
        super(SidaMetaModel, self).__init__(config)

        self.config = config
        if not hasattr(self.config, "train_mask_decoder"):
            self.config.train_mask_decoder = kwargs["train_mask_decoder"]
            self.config.out_dim = kwargs["out_dim"]
            self.vision_pretrained = kwargs.get("vision_pretrained", None)
        else:
            self.vision_pretrained = kwargs.get("vision_pretrained", None)
            self.initialize_sida_modules(self.config)

    def initialize_sida_modules(self, config):
        # SAM
        self.visual_model = build_sam_vit_h(self.vision_pretrained)
        for param in self.visual_model.parameters():
            param.requires_grad = False
        if config.train_mask_decoder:
            self.visual_model.mask_decoder.train()
            for param in self.visual_model.mask_decoder.parameters():
                param.requires_grad = True

        # Projection layer
        in_dim = config.hidden_size
        out_dim = config.out_dim
        text_fc = [
            nn.Linear(in_dim, in_dim),
            nn.ReLU(inplace=True),
            nn.Linear(in_dim, out_dim),
            nn.Dropout(0.0),
        ]
        self.text_hidden_fcs = nn.ModuleList([nn.Sequential(*text_fc)])
        cls_head = (
            nn.Linear(in_dim, in_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.0),
            nn.Linear(in_dim // 2, 3)
        )
        self.cls_head = nn.ModuleList([nn.Sequential(*cls_head)])
        print(f"Created cls_head: {cls_head}")
        self.sida_fc1 = nn.Linear(3, out_dim)
        print(f"Created sida_fc1: {self.sida_fc1}")
        self.attention_layer = nn.MultiheadAttention(embed_dim=out_dim, num_heads=8, batch_first=True)
        print(f"Created attention_layer: {self.attention_layer}")
        self.text_hidden_fcs.train()
        self.cls_head.train()
        self.sida_fc1.train()
        self.attention_layer.train()
        for param in self.text_hidden_fcs.parameters():
            param.requires_grad = True
        for param in self.cls_head.parameters():
            param.requires_grad = True
        for param in self.sida_fc1.parameters():
            param.requires_grad = True
        for param in self.attention_layer.parameters():
            param.requires_grad = True

class SidaModel(SidaMetaModel, LlavaLlamaModel):
    def __init__(self, config, **kwargs):
        super(SidaModel, self).__init__(config, **kwargs)
        
        print("\nInitializing SidaModel:")
        self.config.use_cache = False
        self.config.vision_tower = self.config.mm_vision_tower
        self.config.mm_vision_select_feature = "patch"
        self.config.image_aspect_ratio = "square"
        self.config.image_grid_pinpoints = None
        self.config.tune_mm_mlp_adapter = False
        self.config.freeze_mm_mlp_adapter = True
        self.config.pretrain_mm_mlp_adapter = None
        self.config.mm_use_im_patch_token = False
        self.config.vision_hidden_size = 256
        self.config.fc_hidden_size = 1408
        self.config.llm_input_size = 1024

class SIDAForCausalLM(LlavaLlamaForCausalLM):
    def __init__(self, config, **kwargs):
        if not hasattr(config, "train_mask_decoder"):
            config.mm_use_im_start_end = kwargs.pop("use_mm_start_end", True)
            config.mm_vision_tower = kwargs.get(
                "vision_tower", "openai/clip-vit-large-patch14"
            )
        else:
            config.mm_vision_tower = config.vision_tower

        self.ce_loss_weight = kwargs.pop("ce_loss_weight", None)
        self.dice_loss_weight = kwargs.pop("dice_loss_weight", None)
        self.bce_loss_weight = kwargs.pop("bce_loss_weight", None)
        self.cls_loss_weight = kwargs.pop("cls_loss_weight", None)
        self.mask_loss_weight =  kwargs.pop("mask_loss_weight", None)
        # 2. Initialize base model
        self.cls_token_idx = kwargs.pop("cls_token_idx")
        self.seg_token_idx = kwargs.pop("seg_token_idx")
        super().__init__(config)
        self.model = SidaModel(config, **kwargs)
        self.model.initialize_sida_modules(config)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.post_init()
    def get_visual_embs(self, pixel_values: torch.FloatTensor):
        with torch.no_grad():
            image_embeddings_list = []
            for i in range(pixel_values.shape[0]):
                torch.cuda.empty_cache()
                image_embeddings = self.model.visual_model.image_encoder(
                    pixel_values[i].unsqueeze(0)
                )
                image_embeddings_list.append(image_embeddings)
            torch.cuda.empty_cache()
            image_embeddings = torch.cat(image_embeddings_list, 0)
        return image_embeddings
    
    def forward(self, **kwargs):
        if "past_key_values" in kwargs:
            return super().forward(**kwargs)
        return self.model_forward(**kwargs)
    
    def model_forward(
        self,
        images: torch.FloatTensor,
        images_clip: torch.FloatTensor,
        input_ids: torch.LongTensor,
        cls_labels: torch.LongTensor,  # Classification labels: 0-real, 1-full synthetic, 2-tampered
        labels:torch.LongTensor,
        attention_masks: torch.LongTensor,
        offset: torch.LongTensor,
        masks_list: List[torch.FloatTensor],
        cls_labels_list: List[torch.LongTensor],
        label_list: List[torch.Tensor],
        resize_list: List[tuple],
        inference: bool = False,  # Flag for inference mode
        **kwargs,
    ):
        if images.size(0) != images_clip.size(0):
            raise ValueError(f"Batch size mismatch: images {images.size(0)} != images_clip {images_clip.size(0)}")
        image_embeddings = self.get_visual_embs(images)
        B, C, H, W = image_embeddings.shape
        
        assert B == len(offset) - 1
        cls_token_mask = (input_ids[:,1:] == self.cls_token_idx)
        cls_token_mask = torch.cat([
            cls_token_mask,
            torch.zeros((cls_token_mask.shape[0], 1)).bool().cuda()
            ], 
            dim=1)
        cls_token_mask =  torch.cat(
            [
            torch.zeros((cls_token_mask.shape[0], 255)).bool().cuda(),  # Padding with 255 zeros at the beginning
            cls_token_mask,
            ],
                dim=1,
            )
        seg_token_mask = (input_ids[:, 1:] == self.seg_token_idx)

        seg_token_mask = torch.cat([
            torch.zeros((seg_token_mask.shape[0], 255), dtype=torch.bool, device=input_ids.device),
            seg_token_mask,
            torch.zeros((seg_token_mask.shape[0], 1),   dtype=torch.bool, device=input_ids.device)], dim=1)

        if inference:
            n_batch = 1
            length = input_ids.shape[0]
            assert images_clip.shape[0] == 1
            images_clip_extend = images_clip.expand(length, -1, -1, -1).contiguous()
            output_hidden_states = []
            for i in range(n_batch):
                start_i, end_i = i * length, min((i + 1) * length, input_ids.shape[0])
                output_i = super().forward(
                    images=images_clip_extend[: end_i - start_i],
                    attention_mask=attention_masks[start_i:end_i],
                    input_ids=input_ids[start_i:end_i],
                    output_hidden_states=True,
                )
                output_hidden_states.append(output_i.hidden_states)
                torch.cuda.empty_cache()
            output_hidden_states_list = []
            output_hidden_states_level = torch.cat(output_hidden_states, dim=0)
            output_hidden_states_list.append(output_hidden_states_level)
            output_hidden_states = output_hidden_states_list
            output = None
        else:
            images_clip_list = []
            for i in range(len(offset) - 1):
                start_i, end_i = offset[i], offset[i + 1]
                images_clip_i = (
                    images_clip[i]
                    .unsqueeze(0)
                    .expand(end_i - start_i, -1, -1, -1)
                    .contiguous()
                )
                images_clip_list.append(images_clip_i)
            images_clip = torch.cat(images_clip_list, dim=0) #[2,3,224,224]
            output = super().forward(
                images=images_clip,
                attention_mask=attention_masks,
                input_ids=input_ids,
                labels = labels,
                output_hidden_states=True,
            )
            output_hidden_states = output.hidden_states
            # Geting cls information
        assert len(self.model.cls_head) == 1
        last_hidden_state_cls = self.model.cls_head[0](output_hidden_states[-1]) 

        cls_result = last_hidden_state_cls[cls_token_mask]

        logits = cls_result
        loss_fct = nn.CrossEntropyLoss()
        cls_loss = loss_fct(logits, cls_labels)

        #Geting segmentation
        mask_bce_loss = mask_dice_loss = mask_loss = torch.tensor(0.0, device=cls_loss.device)
        num_masks = 0
        if (cls_labels == 2).any():
            hidden_states = []
            hidden_states.append(self.model.text_hidden_fcs[0](output_hidden_states[-1]))
            last_hidden_state = torch.stack(hidden_states, dim=-1).sum(dim=-1)
            pred_embeddings = last_hidden_state[seg_token_mask]
            seg_token_counts = seg_token_mask.int().sum(-1)  # [bs, ]
            seg_token_offset = seg_token_counts.cumsum(-1)
            seg_token_offset = torch.cat(
            [torch.zeros(1).long().cuda(), seg_token_offset], dim=0
            )
            try:
                seg_token_offset = seg_token_offset[offset]
            except Exception as e:
                print(f"Error when applying offset to seg_token_offset: {e}")
            pred_embeddings_ = []
            for i in range(len(seg_token_offset) - 1):
                start_i, end_i = seg_token_offset[i], seg_token_offset[i + 1]
                pred_embeddings_.append(pred_embeddings[start_i:end_i])
            pred_embeddings = pred_embeddings_
            #Attention
            cls_projected = self.model.sida_fc1(cls_result)
            enhanced_pred_embeddings = []
            for i in range(len(pred_embeddings)):
                seg_embeddings = pred_embeddings[i]
                # Prepare Query, Key, and Value
                query = cls_projected[i].unsqueeze(0)
                key = seg_embeddings
                value = seg_embeddings
                try:
                    attn_output, _ = self.model.attention_layer(query=query, key=key, value=value)
                except Exception as e:
                    print(f"Error in attention layer: {e}")
                enhanced_embeddings = seg_embeddings + attn_output
                enhanced_pred_embeddings.append(enhanced_embeddings)
            multimask_output = False

            pred_masks = []
            for i in range(len(enhanced_pred_embeddings)):
                (
                    
                    sparse_embeddings,
                    dense_embeddings,
                ) = self.model.visual_model.prompt_encoder(
                    points=None,
                    boxes=None,
                    masks=None,
                    text_embeds=enhanced_pred_embeddings[i].unsqueeze(1),
                )


                sparse_embeddings = sparse_embeddings.to(enhanced_pred_embeddings[i].dtype)
                low_res_masks, iou_predictions = self.model.visual_model.mask_decoder(
                    image_embeddings=image_embeddings[i].unsqueeze(0),
                    image_pe=self.model.visual_model.prompt_encoder.get_dense_pe(),
                    sparse_prompt_embeddings=sparse_embeddings,
                    dense_prompt_embeddings=dense_embeddings,
                    multimask_output=multimask_output,
                )


                pred_mask = self.model.visual_model.postprocess_masks(
                    low_res_masks,
                    input_size=resize_list[i],
                    original_size=label_list[i].shape,
                )

                pred_masks.append(pred_mask[:, 0])

            model_output = output
            gt_masks = masks_list

            if inference:
                return {
            "pred_masks": pred_masks,
            "gt_masks": gt_masks,
            "logits": logits,
            }
            for batch_idx in range(len(pred_masks)):
                gt_mask = gt_masks[batch_idx]
                pred_mask = pred_masks[batch_idx]
                assert (
                    gt_mask.shape[0] == pred_mask.shape[0]
                ), "gt_mask.shape: {}, pred_mask.shape: {}".format(
                    gt_mask.shape, pred_mask.shape
                )
                mask_bce_loss += (
                    sigmoid_ce_loss(pred_mask, gt_mask, num_masks=gt_mask.shape[0])
                    * gt_mask.shape[0]
                )
                mask_dice_loss += (
                    dice_loss(pred_mask, gt_mask, num_masks=gt_mask.shape[0])
                    * gt_mask.shape[0]
                )
                num_masks += gt_mask.shape[0]

            mask_bce_loss = self.bce_loss_weight * mask_bce_loss / (num_masks + 1e-8)
            mask_dice_loss = self.dice_loss_weight * mask_dice_loss / (num_masks + 1e-8)
            mask_loss = mask_bce_loss + mask_dice_loss
        else:
            mask_bce_loss = torch.tensor(0.0, device=cls_loss.device)
            mask_dice_loss = torch.tensor(0.0, device=cls_loss.device)
            mask_loss = torch.tensor(0.0, device=cls_loss.device)

        if not inference and seg_token_mask.sum() == 0:  
            dummy = torch.zeros([], device=cls_loss.device) 
            for p in itertools.chain(
                self.model.visual_model.mask_decoder.parameters(),
                self.model.text_hidden_fcs.parameters(),
                self.model.sida_fc1.parameters(),
                self.model.attention_layer.parameters()):
                dummy = dummy + p.sum() * 0.0      
            mask_loss = mask_loss + dummy 

        loss = self.mask_loss_weight * mask_loss + self.cls_loss_weight * cls_loss
        return {
            "loss": loss,
            "mask_bce_loss": mask_bce_loss,
            "mask_dice_loss": mask_dice_loss,
            "mask_loss": mask_loss,
            "cls_loss": cls_loss,
            "logits": logits,
            "cls_hidden_state": cls_result, 
        }

    def evaluate(
        self,
        images_clip,
        images,
        input_ids,
        resize_list,
        original_size_list,
        max_new_tokens=64,
        tokenizer=None,
    ):
        with torch.no_grad():
            # Generate initial output sequence
            outputs = self.generate(
                images=images_clip,
                input_ids=input_ids,
                max_new_tokens=max_new_tokens,
                num_beams=1,
                output_hidden_states=True,
                return_dict_in_generate=True,
            )
            output_hidden_states = outputs.hidden_states[-1]  # Shape: [batch_size, sequence_length, hidden_size]
            output_ids = outputs.sequences  # Generated sequences

            # Assume batch_size=1 for simplicity (as seen in chat.py)
            batch_size = output_ids.shape[0]
            assert batch_size == 1, "Batch size > 1 not handled in this example"

            # Find positions of [CLS] tokens in the sequence
            cls_token_mask = (output_ids[:, 1:] == self.cls_token_idx)
            cls_token_mask = torch.cat(
                [
                    torch.zeros((cls_token_mask.shape[0], 255)).bool().cuda(),
                    cls_token_mask
                ],
                dim=1
            )

            pred_masks = []
            predicted_class = None
            if cls_token_mask.any():
                last_hidden_state_cls = self.model.cls_head[0](output_hidden_states)
                cls_result = last_hidden_state_cls[cls_token_mask]
                if cls_result.size(0) > 0:
                    # Use the last [CLS] token for class prediction
                    last_cls_result = cls_result[-1]
                    predicted_class = torch.argmax(last_cls_result, dim=-1).item()
                    if predicted_class == 2:
                        # Proceed with segmentation if class is tampered
                        seg_token_mask = (output_ids[:, 1:] == self.seg_token_idx)
                        seg_token_mask = torch.cat(
                            [
                                torch.zeros((seg_token_mask.shape[0], 255)).bool().cuda(),
                                seg_token_mask
                            ],
                            dim=1
                        )
                        # Process hidden states for segmentation
                        hidden_states = []
                        hidden_states.append(self.model.text_hidden_fcs[0](output_hidden_states))
                        last_hidden_state = torch.stack(hidden_states, dim=-1).sum(dim=-1)
                        pred_embeddings = last_hidden_state[seg_token_mask]

                        # Process segmentation tokens
                        seg_token_counts = seg_token_mask.int().sum(-1)
                        seg_token_offset = seg_token_counts.cumsum(-1)
                        seg_token_offset = torch.cat(
                            [torch.zeros(1).long().cuda(), seg_token_offset],
                            dim=0
                        )

                        pred_embeddings_ = []
                        for i in range(len(seg_token_offset) - 1):
                            start_i, end_i = seg_token_offset[i], seg_token_offset[i + 1]
                            pred_embeddings_.append(pred_embeddings[start_i:end_i])
                        pred_embeddings = pred_embeddings_

                        # Apply attention mechanism
                        cls_projected = self.model.sida_fc1(cls_result)
                        enhanced_pred_embeddings = []
                        for i in range(len(pred_embeddings)):
                            seg_embeddings = pred_embeddings[i]
                            query = cls_projected[i].unsqueeze(0)
                            key = seg_embeddings
                            value = seg_embeddings
                            try:
                                attn_output, _ = self.model.attention_layer(
                                    query=query,
                                    key=key,
                                    value=value
                                )
                                enhanced_embeddings = seg_embeddings + attn_output
                                enhanced_pred_embeddings.append(enhanced_embeddings)
                            except Exception as e:
                                print(f"Error in attention layer: {e}")
                                enhanced_pred_embeddings.append(seg_embeddings)

                        # Get image embeddings and generate masks
                        image_embeddings = self.get_visual_embs(images)
                        multimask_output = False
                        pred_masks = []
                        for i in range(len(enhanced_pred_embeddings)):
                            sparse_embeddings, dense_embeddings = self.model.visual_model.prompt_encoder(
                                points=None,
                                boxes=None,
                                masks=None,
                                text_embeds=enhanced_pred_embeddings[i].unsqueeze(1),
                            )
                            sparse_embeddings = sparse_embeddings.to(enhanced_pred_embeddings[i].dtype)
                            low_res_masks, iou_predictions = self.model.visual_model.mask_decoder(
                                image_embeddings=image_embeddings[i].unsqueeze(0),
                                image_pe=self.model.visual_model.prompt_encoder.get_dense_pe(),
                                sparse_prompt_embeddings=sparse_embeddings,
                                dense_prompt_embeddings=dense_embeddings,
                                multimask_output=multimask_output,
                            )
                            pred_mask = self.model.visual_model.postprocess_masks(
                                low_res_masks,
                                input_size=resize_list[i],
                                original_size=original_size_list[i],
                            )
                            pred_masks.append(pred_mask[:, 0])

            # Post-process output_ids to ensure correct class description
            if tokenizer is not None and predicted_class is not None:
                # Define class-specific responses
                class_responses = {
                    0: "[CLS] This image is classified as real. It shows no signs of tampering or synthesis.",
                    1: "[CLS] This image is classified as full synthetic. It appears entirely artificially generated.",
                    2: "[CLS] This image is classified as tampered. It has been altered. [SEG] A mask highlighting the tampered region is provided."
                }
                # Tokenize the correct response
                correct_response = class_responses[predicted_class]
                new_output_ids = tokenizer.encode(correct_response, return_tensors="pt").to(output_ids.device)
                # Replace output_ids with the correct tokenized response
                output_ids = new_output_ids

            return output_ids, pred_masks
