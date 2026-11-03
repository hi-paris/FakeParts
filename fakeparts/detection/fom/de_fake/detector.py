from __future__ import annotations

from pathlib import Path
from ...base import FrameDetector


class DeFake(FrameDetector):
    """DE-FAKE image-caption and CLIP-feature fusion classifier."""

    def __init__(self, ckpt_path, device="cuda", threshold=0.5, crop_size=None,
                 work_dir=None, ckpt_paths=None, version="default"):
        super().__init__(ckpt_path, device, threshold, crop_size, work_dir, ckpt_paths, version)
        if len(self.ckpt_paths) != 3:
            raise ValueError("De-Fake requires classifier, fine-tuned CLIP, and BLIP caption weights")

        import sys
        import types
        import torch
        from torchvision import transforms

        project_root = Path(__file__).resolve().parents[4]
        optional_deps = project_root / ".deps" / "detection-extra"
        if optional_deps.is_dir() and str(optional_deps) not in sys.path:
            # Append optional deps so they supply packages absent from the
            # active runtime without shadowing its CUDA-enabled torch stack.
            sys.path.append(str(optional_deps))
        detection_deps = project_root / ".deps" / "detection"
        if detection_deps.is_dir() and str(detection_deps) not in sys.path:
            sys.path.append(str(detection_deps))
        clip_root = project_root / "fakeparts/detection/fom/d3/models"
        lavis_root = project_root / "fakeparts/detection/vlm/afprompt"
        for root in (str(clip_root), str(lavis_root)):
            if root not in sys.path:
                sys.path.insert(0, root)

        # LAVIS's top-level packages eagerly import every model family. De-Fake
        # only needs BLIP captioning, so expose the vendored package paths and
        # import those modules directly without initializing unrelated models.
        lavis_package = types.ModuleType("lavis")
        lavis_package.__path__ = [str(lavis_root / "lavis")]
        lavis_package.__package__ = "lavis"
        sys.modules.setdefault("lavis", lavis_package)
        models_package = types.ModuleType("lavis.models")
        models_package.__path__ = [str(lavis_root / "lavis" / "models")]
        models_package.__package__ = "lavis.models"
        sys.modules.setdefault("lavis.models", models_package)
        from lavis.models.base_model import BaseModel
        models_package.BaseModel = BaseModel
        blip_package = types.ModuleType("lavis.models.blip_models")
        blip_package.__path__ = [str(lavis_root / "lavis" / "models" / "blip_models")]
        blip_package.__package__ = "lavis.models.blip_models"
        sys.modules.setdefault("lavis.models.blip_models", blip_package)
        from lavis.common.registry import registry
        registry.register_path("library_root", str(lavis_root / "lavis"))
        registry.register_path("repo_root", str(lavis_root))

        from clip import tokenize
        from clip.clip import _transform, build_model
        import transformers.modeling_utils as modeling_utils
        from transformers import pytorch_utils
        for helper in (
            "apply_chunking_to_forward",
            "prune_linear_layer",
        ):
            if not hasattr(modeling_utils, helper) and hasattr(pytorch_utils, helper):
                setattr(modeling_utils, helper, getattr(pytorch_utils, helper))
        if not hasattr(modeling_utils, "find_pruneable_heads_and_indices"):
            def find_pruneable_heads_and_indices(heads, n_heads, head_size, already_pruned_heads):
                heads = set(heads) - set(already_pruned_heads)
                mask = torch.ones(n_heads, head_size)
                for head in heads:
                    adjusted = head - sum(pruned < head for pruned in already_pruned_heads)
                    mask[adjusted] = 0
                flat_mask = mask.view(-1).contiguous().eq(1)
                index = torch.arange(len(flat_mask))[flat_mask].long()
                return heads, index
            modeling_utils.find_pruneable_heads_and_indices = find_pruneable_heads_and_indices
            pytorch_utils.find_pruneable_heads_and_indices = find_pruneable_heads_and_indices
        try:
            import transformers.utils.model_parallel_utils  # noqa: F401
        except ImportError:
            model_parallel_utils = types.ModuleType("transformers.utils.model_parallel_utils")

            def get_device_map(n_layers, devices):
                devices = list(devices)
                if not devices:
                    raise ValueError("model parallelism requires at least one device")
                quotient, remainder = divmod(n_layers, len(devices))
                mapping, start = {}, 0
                for index, device_id in enumerate(devices):
                    count = quotient + (index < remainder)
                    if count:
                        mapping[device_id] = list(range(start, start + count))
                        start += count
                return mapping

            def assert_device_map(device_map, n_blocks):
                blocks = [block for indices in device_map.values() for block in indices]
                if sorted(blocks) != list(range(n_blocks)):
                    raise ValueError("device_map must assign every model block exactly once")

            model_parallel_utils.get_device_map = get_device_map
            model_parallel_utils.assert_device_map = assert_device_map
            sys.modules["transformers.utils.model_parallel_utils"] = model_parallel_utils
        from lavis.common.config import OmegaConf
        from transformers import PreTrainedModel
        # LAVIS's bundled BERT predates Transformers' renamed tie-weight
        # mapping. Current Transformers exposes this property but returns None
        # for models without tied keys; its implementation expects a mapping.
        PreTrainedModel.all_tied_weights_keys = property(
            lambda model: getattr(model, "_tied_weights_keys", None) or {}
        )
        from lavis.models.blip_models.blip_caption import BlipCaption
        from lavis.models.med import BertPreTrainedModel
        from lavis.models.med import XBertLMHeadDecoder
        from transformers.generation import GenerationMixin
        from transformers import __version__ as transformers_version

        # LAVIS's vendored BERT implementation uses mask helpers that were
        # removed from PreTrainedModel in Transformers 5.x.
        if not hasattr(BertPreTrainedModel, "invert_attention_mask"):
            def invert_attention_mask(model, attention_mask):
                if attention_mask.dim() == 3:
                    extended = attention_mask[:, None, :, :]
                elif attention_mask.dim() == 2:
                    extended = attention_mask[:, None, None, :]
                else:
                    raise ValueError(
                        "attention_mask must have 2 or 3 dimensions; "
                        f"got shape {tuple(attention_mask.shape)}"
                    )
                extended = extended.to(dtype=model.dtype)
                return (1.0 - extended) * torch.finfo(model.dtype).min

            def convert_head_mask_to_5d(model, head_mask, num_hidden_layers):
                if head_mask.dim() == 1:
                    head_mask = head_mask[None, None, :, None, None]
                    head_mask = head_mask.expand(num_hidden_layers, -1, -1, -1, -1)
                elif head_mask.dim() == 2:
                    head_mask = head_mask[:, None, :, None, None]
                if head_mask.dim() != 5:
                    raise ValueError(f"head_mask must expand to 5 dimensions; got {head_mask.dim()}")
                return head_mask.to(dtype=model.dtype)

            def get_head_mask(model, head_mask, num_hidden_layers, is_attention_chunked=False):
                if head_mask is None:
                    return [None] * num_hidden_layers
                head_mask = convert_head_mask_to_5d(model, head_mask, num_hidden_layers)
                return head_mask.unsqueeze(-1) if is_attention_chunked else head_mask

            BertPreTrainedModel.invert_attention_mask = invert_attention_mask
            BertPreTrainedModel.get_head_mask = get_head_mask

        BertPreTrainedModel.all_tied_weights_keys = property(
            lambda model: getattr(model, "_tied_weights_keys", None) or {}
        )
        # Transformers 4.50 removed GenerationMixin from PreTrainedModel's
        # inheritance chain. This vendored BLIP decoder relies on generate(),
        # so restore the mixin on its actual upstream decoder class.
        if not issubclass(XBertLMHeadDecoder, GenerationMixin):
            XBertLMHeadDecoder.__bases__ = (*XBertLMHeadDecoder.__bases__, GenerationMixin)
        # Transformers 5 expands tensor-valued model kwargs to beam count inside
        # generate(). LAVIS 0.2 also expands visual_embeds before calling it,
        # which duplicates the beam dimension and mismatches the decoder batch.
        if int(transformers_version.split(".", 1)[0]) >= 5:
            def generate_from_encoder_v5(
                model, tokenized_prompt, visual_embeds, sep_token_id, pad_token_id,
                use_nucleus_sampling=False, num_beams=3, max_length=30,
                min_length=10, top_p=0.9, repetition_penalty=1.0, **kwargs,
            ):
                image_atts = torch.ones(
                    visual_embeds.size()[:-1], dtype=torch.long, device=model.device
                )
                model_kwargs = {
                    "encoder_hidden_states": visual_embeds,
                    "encoder_attention_mask": image_atts,
                }
                common = {
                    "input_ids": tokenized_prompt.input_ids,
                    "max_length": max_length,
                    "min_length": min_length,
                    "eos_token_id": sep_token_id,
                    "pad_token_id": pad_token_id,
                    "repetition_penalty": repetition_penalty,
                    **model_kwargs,
                    **kwargs,
                }
                if use_nucleus_sampling:
                    return model.generate(
                        **common, do_sample=True, top_p=top_p,
                        num_return_sequences=1,
                    )
                return model.generate(**common, num_beams=num_beams)

            XBertLMHeadDecoder.generate_from_encoder = generate_from_encoder_v5
        from lavis.models.blip_models.blip import BlipBase
        from lavis.models.base_model import BaseModel
        # The vendored LAVIS release rejects Transformers >=4.27 at init even
        # though the adapter supplies its legacy BERT compatibility shims.
        BlipBase.__init__ = BaseModel.__init__
        from transformers import BertTokenizer

        def compatible_init_tokenizer(cls):
            tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
            tokenizer.add_special_tokens({"bos_token": "[DEC]"})
            tokenizer.add_special_tokens({"additional_special_tokens": ["[ENC]"]})
            tokenizer.enc_token_id = tokenizer.convert_tokens_to_ids("[ENC]")
            return tokenizer

        BlipBase.init_tokenizer = classmethod(compatible_init_tokenizer)

        self.torch = torch
        self.tokenize = tokenize
        clip_state = torch.load(self.ckpt_paths[1], map_location="cpu", weights_only=False)
        if isinstance(clip_state, torch.nn.Module):
            self.clip = clip_state.to(device)
            self.clip_preprocess = _transform(self.clip.visual.input_resolution)
        else:
            if isinstance(clip_state, dict) and "state_dict" in clip_state:
                clip_state = clip_state["state_dict"]
            self.clip = build_model(clip_state).to(device)
            self.clip_preprocess = _transform(self.clip.visual.input_resolution)
        self.clip.eval()

        # The reused D3 CLIP fork returns (intermediate_features, tokens) from
        # Transformer.forward(), while its encode_text() still assumes the
        # original tensor-only OpenAI CLIP API. De-Fake only needs the final
        # text embedding, so unwrap the token tensor locally for this adapter.
        from types import MethodType

        def encode_text_compatible(model, text):
            tokens = model.token_embedding(text).type(model.dtype)
            tokens = tokens + model.positional_embedding.type(model.dtype)
            tokens = tokens.permute(1, 0, 2)
            transformed = model.transformer(tokens)
            if isinstance(transformed, tuple):
                _, tokens = transformed
            else:
                tokens = transformed
            tokens = tokens.permute(1, 0, 2)
            tokens = model.ln_final(tokens).type(model.dtype)
            return tokens[torch.arange(tokens.shape[0]), text.argmax(dim=-1)] @ model.text_projection

        self.clip.encode_text = MethodType(encode_text_compatible, self.clip)

        cfg_path = Path(BlipCaption.default_config_path("base_coco"))
        cfg = OmegaConf.load(cfg_path).model
        cfg.image_size = 224
        cfg.load_finetuned = False
        cfg.load_pretrained = True
        cfg.pretrained = str(self.ckpt_paths[2])
        self.captioner = BlipCaption.from_config(cfg).to(device).eval()
        self.caption_transform = transforms.Compose([
            transforms.Resize(224),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
        ])

        # The released classifier checkpoint was serialized from a training
        # script whose class lived at ``__main__.NeuralNet``.
        import __main__
        if not hasattr(__main__, "NeuralNet"):
            __main__.NeuralNet = self._head_class(torch)
        state = torch.load(self.ckpt_paths[0], map_location="cpu", weights_only=False)
        if isinstance(state, torch.nn.Module):
            self.classifier = state.to(device).eval()
        else:
            if isinstance(state, dict) and "state_dict" in state:
                state = state["state_dict"]
            self.classifier = self._build_head(torch, state).to(device).eval()

    @staticmethod
    def _head_class(torch):
        import torch.nn.functional as F
        from torch import nn

        class NeuralNet(nn.Module):
            def __init__(self):
                super().__init__()
                self.dropout2 = nn.Dropout(0.5)
                self.fc1 = nn.Linear(1024, 512)
                self.fc2 = nn.Linear(512, 256)
                self.fc3 = nn.Linear(256, 2)

            def forward(self, features):
                features = F.relu(self.fc1(features))
                features = self.dropout2(features)
                features = F.relu(self.fc2(features))
                return self.fc3(features)

        return NeuralNet

    @staticmethod
    def _build_head(torch, state):
        if not isinstance(state, dict):
            raise TypeError(f"unsupported De-Fake classifier checkpoint: {type(state).__name__}")
        state = {key.removeprefix("module."): value for key, value in state.items()}
        model = DeFake._head_class(torch)()
        model.load_state_dict(state)
        return model

    def predict_frames(self, images):
        torch = self.torch
        clip_images = torch.stack([self.clip_preprocess(image.convert("RGB")) for image in images])
        caption_images = torch.stack([self.caption_transform(image.convert("RGB")) for image in images])
        with torch.inference_mode():
            captions = self.captioner.generate(
                {"image": caption_images.to(self.device)},
                use_nucleus_sampling=False, num_beams=3, max_length=60, min_length=5,
            )
            tokens = self.tokenize(captions).to(self.device)
            image_features = self.clip.encode_image(clip_images.to(self.device))
            text_features = self.clip.encode_text(tokens)
            fused = torch.cat((image_features, text_features), dim=1).float()
            logits = self.classifier(fused)
            scores = logits.softmax(dim=1)[:, 1].cpu().tolist()
        return [self.result(score) for score in scores]
