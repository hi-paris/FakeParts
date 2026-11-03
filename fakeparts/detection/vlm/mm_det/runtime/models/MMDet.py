import torch
import numpy as np
from PIL import Image
from torch import nn
from einops import rearrange
from transformers import AutoProcessor
import os

from .vit.stv_transformer_hybrid import vit_base_r50_s16_224_with_recons_iafa
from llava.constants import IMAGE_TOKEN_INDEX
from llava.model.builder import load_pretrained_model as load_llava_pretrained_model
from llava.mm_utils import process_images, tokenizer_image_token, get_model_name_from_path


class MMEncoder(nn.Module):
    def __init__(self, config, **kwargs):
        super().__init__()
        self.model_path = config['lmm_ckpt']
        self.model_base = config['lmm_base']
        self.model_name = get_model_name_from_path(self.model_path)
        self.load_8bit = config['load_8bit']
        self.load_4bit = config['load_4bit']
        self.tokenizer, self.model, self.image_processor, self.context_len = load_llava_pretrained_model(model_path=self.model_path, model_base=self.model_base, load_8bit=self.load_8bit, load_4bit=self.load_4bit, model_name=self.model_name)
        self.vision_tower = self.model.get_vision_tower().vision_tower
        vision_tower_config_path = getattr(self.vision_tower.config, "_name_or_path")
        self.visual_processor = AutoProcessor.from_pretrained(vision_tower_config_path)
        self.conv_mode = config['conv_mode']
        self.new_tokens = config['new_tokens']
        self.selected_layers = config['selected_layers']
        self.interval = config['interval']
        
    def get_prompt(self):
        if self.conv_mode == 'llava_v1':
            assistant_intro = "Assistant: A chat between a curious human and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the human's questions.###"
            human_instruction = "Human: <image>\nAs an expert in image forensics, you are to briefly describe the image, including lighting and reflection, texture, color saturation, shape consistency, sense of depth, compression trace, artifacts. Give a reason to justify whether it is a real or a fake image.###"
            assistant_response = "Assistant:"
            text = assistant_intro + human_instruction + assistant_response
        elif self.conv_mode == 'mistral':
            text = "[INST] <image>\nAs an expert in image forensics, you are to briefly describe the image, including lighting and reflection, texture, color saturation, shape consistency, sense of depth, compression trace, artifacts. Give a reason to justify whether it is a real or a fake image. [/INST]"
        else:
            raise ValueError(f'Expect conv mode in "llava_v1", "mistral". Get {self.conv_mode}.')
        return text
    
    def encode_visual_features(self, images, image_sizes=None):
        visual_input = self.visual_processor(images=images, return_tensors="pt").to(self.vision_tower.device)
        visual_input['pixel_values'] = visual_input['pixel_values'].half()
        clip_features = self.model.get_vision_tower().vision_tower(**visual_input).pooler_output
        return clip_features
            
    def forward(self, x):
        with torch.inference_mode():
            images = []
            image_sizes = []
            if isinstance(x, torch.Tensor):
                if len(x.shape) == 4:    # add timesteps if an image is forwarded
                    x = x.unsqueeze(0)
                try:
                    assert(x.size(1) <= self.interval)
                except:
                    raise ValueError(f'The video length {x.size(1)} is longer than the maximum interval {self.interval} for a single inference. Please divide input into more short clips.')
                x = x[:, 0, :, :, :].squeeze(1)    # inference for the first frame
                x = rearrange(x, 'b c h w -> b h w c')
                for t in x:
                    img = Image.fromarray((t.cpu().numpy() * 255).astype(np.uint8))
                    images.append(img)
                    image_sizes.append(img.size)
            elif isinstance(x, Image.Image):
                images.append(x)
                image_sizes.append(x.size)
            else:
                raiseValueError(f'Unsupported image type: {type(x)}')
                
            image_t = process_images(images, self.image_processor, self.model.config)
            if type(image_t) is list:
                image_t = [image.to(self.model.device, dtype=torch.float16) for image in image_t]
            else:
                image_t = [image_t.to(self.model.device, dtype=torch.float16)]
            prompt = self.get_prompt()
            input_ids = tokenizer_image_token(prompt, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).to(self.model.device)
            visual_features = self.encode_visual_features(images=images, image_sizes=image_sizes)
            textual_features = []
            for idx, t in enumerate(image_t[0]):
                output = self.model.generate(
                    input_ids,
                    images=t.unsqueeze(0),
                    image_sizes=[image_sizes[idx]],
                    do_sample=False,
                    min_new_tokens=self.new_tokens,
                    max_new_tokens=self.new_tokens,
                    use_cache=True,
                    output_hidden_states=True,
                    return_dict_in_generate=True)
                hidden_features = []
                for i in self.selected_layers:
                    for hs in output['hidden_states']:
                        hidden_features.append(hs[i])
                    hidden_feature = torch.cat(hidden_features, dim=1)
                    new_token_feature = hidden_feature[:, hidden_feature.size(1) - self.new_tokens:, :]
                    textual_features.append(new_token_feature)
            textual_features = torch.cat(textual_features, dim=0)
        return visual_features.clone(), textual_features.clone()
    

class Attention(nn.Module):
    def __init__(
            self,
            dim: int,
            num_heads: int = 8,
            attn_drop: float = 0.,
            proj_drop: float = 0.,
            norm_layer: nn.Module = nn.LayerNorm,
            out_dim: int = 1
    ) -> None:
        super().__init__()
        assert dim % num_heads == 0
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=True)
        self.q_norm = nn.Identity()
        self.k_norm = nn.Identity()
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        q, k = self.q_norm(q), self.k_norm(k)
        q = q * self.scale
        attn = q @ k.transpose(-2, -1)
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        x = attn @ v
        x = x.transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        return x
    
    
class DynamicFusion(nn.Module):
    def __init__(self, in_planes):
        super(DynamicFusion, self).__init__()
        self.channel_attn = Attention(dim=in_planes, num_heads=8)
    
    def forward(self, x, output_weights=False):
        cw_weights = self.channel_attn(x)
        x = x * cw_weights.expand_as(x)
        if output_weights:
            out = x, cw_weights
        else:
            out = x
        return out
    
    
class MMDet(nn.Module):
    def __init__(self, config, **kwargs):
        super(MMDet, self).__init__()
        self.window_size = config['window_size']
        self.st_pretrained = config['st_pretrained']
        self.st_ckpt = config['st_ckpt']
        self.lmm_ckpt = config['lmm_ckpt']
        if (not self.st_ckpt or not os.path.exists(self.st_ckpt)) and config['st_pretrained']:
            print('Local pretrained checkpoint for Hybrid ViT not found. Using the default interface in timm.')
            self.st_ckpt = None
        self.backbone = vit_base_r50_s16_224_with_recons_iafa(window_size=config['window_size'], pretrained=config['st_pretrained'], ckpt_path=self.st_ckpt)
        self.load_mm_encoder = not config['cache_mm']
        if self.load_mm_encoder:
            self.mm_encoder = MMEncoder(config)
            for m in self.mm_encoder.modules():
                m.required_grad = False
            print('Freeze MM Encoder.')
        self.clip_proj = nn.Linear(1024, 768)
        self.mm_proj = nn.Linear(4096, 768)
        self.final_fusion = DynamicFusion(in_planes=768)
        self.head = nn.Linear(768, 2)
                
        new_component_list = [self.clip_proj, self.mm_proj, self.final_fusion, self.head]
        for component in new_component_list:
            for m in component.modules():
                if isinstance(m, nn.Conv1d):
                    nn.init.kaiming_normal_(m.weight)
                elif isinstance(m, nn.Linear):
                    nn.init.normal_(m.weight, std=0.01)
                    
    def forward(self, x_input, cached_features={}):
        x_original, x_recons = x_input
        B = x_original.size(0)
        x_st = self.backbone(x_input)    # spatial temporal feature
        visual_feat = cached_features.get('visual', None)
        textual_feat = cached_features.get('textual', None)
        if not self.load_mm_encoder:
            assert(visual_feat is not None and textual_feat is not None)
        if visual_feat is None or textual_feat is None:
            visual_feat, textual_feat = self.mm_encoder(x_original)
        visual_feat, textual_feat = visual_feat.float(), textual_feat.float()
        x_visual = self.clip_proj(visual_feat).unsqueeze(1)
        x_mm = self.mm_proj(textual_feat)
        x_feat = torch.cat([x_st, x_visual, x_mm], dim=1)
        x_feat = self.final_fusion(x_feat)
        x_feat = torch.mean(x_feat, dim=1)
        out = self.head(x_feat)
        return out
    