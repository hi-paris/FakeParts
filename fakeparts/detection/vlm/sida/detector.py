from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Sequence

import numpy as np
from PIL import Image

from fakeparts.detection.base import FrameDetector


class SIDA(FrameDetector):
    """SIDA classification using the authors' image prompt and [CLS] output."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._load()

    def _load(self) -> None:
        import torch
        import torch.nn.functional as F
        from transformers import AutoTokenizer, CLIPImageProcessor

        repo = Path(__file__).parent
        sys.path.insert(0, str(repo))
        try:
            from model.SIDA import SIDAForCausalLM
            from model.llava import conversation as conversation_lib
            from model.llava.mm_utils import tokenizer_image_token
            from model.segment_anything.utils.transforms import ResizeLongestSide
            from utils.utils import (DEFAULT_IMAGE_TOKEN, DEFAULT_IM_END_TOKEN,
                                     DEFAULT_IM_START_TOKEN, IMAGE_TOKEN_INDEX)
        except ImportError as exc:
            raise RuntimeError(
                "SIDA needs its optional requirements from "
                "fakeparts/detection/vlm/sida/requirements.txt and the authors' CUDA stack."
            ) from exc

        if not torch.cuda.is_available():
            raise RuntimeError("SIDA's released inference path requires CUDA.")
        self.torch = torch
        self.F = F
        self.tokenizer_image_token = tokenizer_image_token
        self.ResizeLongestSide = ResizeLongestSide
        self.image_processor_cls = CLIPImageProcessor
        self.constants = (DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN,
                          DEFAULT_IM_END_TOKEN, IMAGE_TOKEN_INDEX)
        self.conversation = conversation_lib
        tokenizer = AutoTokenizer.from_pretrained(
            self.ckpt_path, model_max_length=512, padding_side="right", use_fast=False
        )
        tokenizer.pad_token = tokenizer.unk_token
        seg_token_idx = tokenizer("[SEG]", add_special_tokens=False).input_ids[0]
        cls_token_idx = tokenizer("[CLS]", add_special_tokens=False).input_ids[0]
        dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        model = SIDAForCausalLM.from_pretrained(
            self.ckpt_path, low_cpu_mem_usage=True,
            vision_tower="openai/clip-vit-large-patch14",
            vision_pretrained=str(self.ckpt_paths[1]), train_mask_decoder=True,
            out_dim=256, use_mm_start_end=True,
            seg_token_idx=seg_token_idx, cls_token_idx=cls_token_idx,
            torch_dtype=dtype,
        ).cuda()
        model.config.eos_token_id = tokenizer.eos_token_id
        model.config.bos_token_id = tokenizer.bos_token_id
        model.config.pad_token_id = tokenizer.pad_token_id
        model.get_model().initialize_vision_modules(model.get_model().config)
        model.get_model().get_vision_tower().to(device="cuda", dtype=dtype)
        self.dtype = dtype
        self.tokenizer = tokenizer
        self.model = model.eval()
        self.image_processor = CLIPImageProcessor.from_pretrained(model.config.vision_tower)
        self.resize = ResizeLongestSide(1024)
        self.image_token_index = IMAGE_TOKEN_INDEX
        self.cls_token_idx = cls_token_idx

    def _one(self, image: Image.Image) -> dict[str, Any]:
        torch = self.torch
        F = self.F
        DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN, _ = self.constants
        conv = self.conversation.conv_templates["llava_v1"].copy()
        prompt = (
            DEFAULT_IMAGE_TOKEN
            + "\nPlease answer begin with [CLS] for classification, if the image is "
            "tampered, output mask the tampered region."
        )
        prompt = prompt.replace(
            DEFAULT_IMAGE_TOKEN,
            DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN,
        )
        conv.append_message(conv.roles[0], prompt)
        conv.append_message(conv.roles[1], "")
        input_prompt = conv.get_prompt()

        image_np = np.asarray(image.convert("RGB"))
        original_size = image_np.shape[:2]
        clip_image = self.image_processor.preprocess(image_np, return_tensors="pt")[
            "pixel_values"
        ].cuda().to(self.dtype)
        resized = self.resize.apply_image(image_np)
        x = torch.from_numpy(resized).permute(2, 0, 1).contiguous().float()
        mean = torch.tensor([123.675, 116.28, 103.53]).view(3, 1, 1)
        std = torch.tensor([58.395, 57.12, 57.375]).view(3, 1, 1)
        x = (x - mean) / std
        x = F.pad(x, (0, 1024 - x.shape[-1], 0, 1024 - x.shape[-2]))
        x = x.unsqueeze(0).cuda().to(self.dtype)
        tokens = self.tokenizer_image_token(
            input_prompt, self.tokenizer, self.image_token_index,
            return_tensors="pt",
        ).unsqueeze(0).cuda()

        with torch.inference_mode():
            generated = self.model.generate(
                # The detector reads only the hidden state at the first [CLS]
                # token; generating the authors' explanatory continuation is
                # unnecessary for classification and dominates per-frame time.
                images=clip_image, input_ids=tokens, max_new_tokens=1,
                num_beams=1, output_hidden_states=True,
                return_dict_in_generate=True,
            )
        output_ids = generated.sequences
        cls_token_positions = output_ids[:, 1:] == self.cls_token_idx
        if not cls_token_positions.any():
            raise RuntimeError("SIDA did not emit its required [CLS] classification token.")
        # Transformers 5 returns only the last decoding step in
        # `generate().hidden_states[-1]`; SIDA's evaluator expects hidden
        # states for the complete sequence (including the 255 visual-token
        # expansion). Re-run the generated sequence with caching disabled to
        # recover the aligned token states used by the released evaluator.
        with torch.inference_mode():
            prepared = self.model.prepare_inputs_labels_for_multimodal(
                input_ids=output_ids,
                attention_mask=None,
                past_key_values=None,
                labels=None,
                images=clip_image,
            )
            prepared_ids, attention_mask, past_key_values, inputs_embeds, _ = prepared
            sequence_outputs = self.model.model(
                input_ids=prepared_ids,
                attention_mask=attention_mask,
                past_key_values=past_key_values,
                inputs_embeds=inputs_embeds,
                output_hidden_states=True,
                use_cache=False,
                return_dict=True,
            )
        output_hidden_states = sequence_outputs.hidden_states[-1]
        # Replacing the image marker with visual tokens changes the sequence
        # length. The upstream code hard-codes a 255-token prefix, but the
        # released CLIP encoder in this configuration expands to 256 tokens.
        # Derive the offset from the actual model output so [CLS] states and
        # logits stay aligned across Transformers/model revisions.
        visual_prefix = output_hidden_states.shape[1] - cls_token_positions.shape[1]
        if visual_prefix < 0:
            raise RuntimeError(
                "SIDA hidden-state sequence is shorter than the generated token sequence."
            )
        cls_token_mask = torch.cat(
            [
                torch.zeros(
                    (cls_token_positions.shape[0], visual_prefix), dtype=torch.bool,
                    device=cls_token_positions.device,
                ),
                cls_token_positions,
            ],
            dim=1,
        )
        with torch.inference_mode():
            cls_logits = self.model.model.cls_head[0](output_hidden_states)[cls_token_mask]
        if not cls_logits.shape[0]:
            raise RuntimeError("SIDA did not expose hidden states aligned to [CLS].")
        logits = cls_logits[-1]
        probs = logits.float().softmax(dim=-1)
        # Author labels: 0 real, 1 full synthetic, 2 tampered.
        fake_score = float((probs[1] + probs[2]).item())
        return self.result(fake_score) | {
            "class_probabilities": [float(v) for v in probs.cpu()],
            "class_id": int(probs.argmax().item()),
            "class_names": ["real", "full_synthetic", "tampered"],
            "input_size": list(original_size),
        }

    def predict_frames(self, img_list: Sequence[Any]) -> list[dict[str, Any]]:
        return [self._one(image) for image in img_list]
