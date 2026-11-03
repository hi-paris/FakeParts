from __future__ import annotations

from pathlib import Path

from ...base import FrameDetector


class C2P(FrameDetector):
    """C2P-CLIP image classifier using its released CLIP linear-head weights."""

    def __init__(self, ckpt_path, device="cuda", threshold=0.5, crop_size=None,
                 work_dir=None, ckpt_paths=None, version="default"):
        super().__init__(ckpt_path, device, threshold, crop_size, work_dir, ckpt_paths, version)
        import torch
        from torch import nn
        from torchvision import transforms
        from transformers import CLIPConfig, CLIPModel, CLIPVisionConfig

        class C2PHead(nn.Module):
            def __init__(self):
                super().__init__()
                # Keep the released state-dict namespace (`model.*`) intact.
                # The C2P release checkpoint contains the complete CLIP
                # ViT-L/14 vision tower and projection weights, so construct
                # its architecture directly instead of relying on a separate
                # Hugging Face cache entry.
                vision_config = CLIPVisionConfig(
                    hidden_size=1024,
                    intermediate_size=4096,
                    num_hidden_layers=24,
                    num_attention_heads=16,
                    image_size=224,
                    patch_size=14,
                    hidden_act="quick_gelu",
                    layer_norm_eps=1e-5,
                )
                self.model = CLIPModel(CLIPConfig(
                    vision_config=vision_config,
                    projection_dim=768,
                ))
                del self.model.text_model
                del self.model.text_projection
                del self.model.logit_scale
                self.model.vision_model.requires_grad_(False)
                self.model.visual_projection.requires_grad_(False)
                self.model.fc = nn.Linear(768, 1)

            def forward(self, images):
                output = self.model.vision_model(
                    pixel_values=images,
                    output_attentions=self.model.config.output_attentions,
                    output_hidden_states=self.model.config.output_hidden_states,
                    return_dict=self.model.config.use_return_dict,
                )
                features = self.model.visual_projection(output[1])
                features = features / features.norm(p=2, dim=-1, keepdim=True)
                return self.model.fc(features)

        state = torch.load(Path(ckpt_path), map_location="cpu", weights_only=False)
        if isinstance(state, dict):
            state.pop("model.vision_model.embeddings.position_ids", None)
        self.model = C2PHead()
        self.model.load_state_dict(state, strict=False)
        self.model.to(device).eval()
        size = crop_size or 224
        self.transform = transforms.Compose([
            transforms.Resize(224, interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.CenterCrop(size),
            transforms.ToTensor(),
            transforms.Normalize(
                [0.48145466, 0.4578275, 0.40821073],
                [0.26862954, 0.26130258, 0.27577711],
            ),
        ])

    def predict_frames(self, images):
        import torch

        inputs = torch.stack([self.transform(image.convert("RGB")) for image in images])
        with torch.inference_mode():
            scores = self.model(inputs.to(self.device)).sigmoid().flatten().cpu().tolist()
        return [self.result(score) for score in scores]
