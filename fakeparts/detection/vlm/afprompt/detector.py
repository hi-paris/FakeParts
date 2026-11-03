from __future__ import annotations

import importlib
import os
import sys
import types
from pathlib import Path
from typing import Sequence

from ...base import FrameDetector


class AFPrompt(FrameDetector):
    """AntifakePrompt InstructBLIP yes/no image detector."""

    question = "Is this photo real [*]?"
    candidates = ("yes", "no")

    def __init__(self, ckpt_path, device="cuda", threshold=0.5, crop_size=None,
                 work_dir=None, ckpt_paths=None, version="default"):
        super().__init__(ckpt_path, device, threshold, crop_size, work_dir, ckpt_paths, version)
        if crop_size is not None and crop_size != 224:
            raise ValueError(f"AntifakePrompt uses 224px inputs; got crop_size={crop_size}")
        if len(self.ckpt_paths) < 2:
            raise ValueError("AntifakePrompt requires a fine-tuned prompt checkpoint and BLIP2 base weights")

        vendor_root = Path(__file__).resolve().parent
        if str(vendor_root) not in sys.path:
            sys.path.insert(0, str(vendor_root))

        # LAVIS eagerly imports every model, task, and dataset from its package
        # initializers. AFPrompt only needs BLIP-2 Vicuna and one image
        # processor; load those modules without importing unrelated optional
        # packages such as spaCy and pycocoevalcap.
        lavis_root = vendor_root / "lavis"
        lavis_package = types.ModuleType("lavis")
        lavis_package.__path__ = [str(lavis_root)]
        lavis_package.__package__ = "lavis"
        sys.modules["lavis"] = lavis_package
        models_package = types.ModuleType("lavis.models")
        models_package.__path__ = [str(lavis_root / "models")]
        models_package.__package__ = "lavis.models"
        sys.modules["lavis.models"] = models_package
        processors_package = types.ModuleType("lavis.processors")
        processors_package.__path__ = [str(lavis_root / "processors")]
        processors_package.__package__ = "lavis.processors"
        sys.modules["lavis.processors"] = processors_package

        import torch
        import transformers.modeling_utils as modeling_utils
        import transformers.pytorch_utils as pytorch_utils
        import transformers.models.clip.modeling_clip as clip_modeling
        import transformers.utils as transformers_utils
        from omegaconf import OmegaConf

        if not hasattr(modeling_utils, "apply_chunking_to_forward"):
            modeling_utils.apply_chunking_to_forward = pytorch_utils.apply_chunking_to_forward

        def find_pruneable_heads_and_indices(heads, n_heads, head_size, already_pruned_heads):
            heads = set(heads) - already_pruned_heads
            mask = torch.ones(n_heads, head_size)
            for head in heads:
                adjusted_head = head - sum(pruned < head for pruned in already_pruned_heads)
                mask[adjusted_head] = 0
            indices = torch.arange(mask.numel())[mask.view(-1).contiguous().eq(1)]
            return heads, indices

        def prune_linear_layer(layer, index, dim=0):
            weight = layer.weight.index_select(dim, index).detach().clone()
            if dim == 1:
                weight = weight.t().contiguous()
            bias = None
            if layer.bias is not None:
                bias = (layer.bias.index_select(0, index).detach().clone()
                        if dim == 0 else layer.bias.detach().clone())
            new_layer = torch.nn.Linear(weight.size(1), weight.size(0), bias=bias is not None)
            new_layer.to(device=layer.weight.device, dtype=layer.weight.dtype)
            new_layer.weight.copy_(weight)
            if bias is not None:
                new_layer.bias.copy_(bias)
            return new_layer

        for module in (modeling_utils, pytorch_utils):
            if not hasattr(module, "find_pruneable_heads_and_indices"):
                module.find_pruneable_heads_and_indices = find_pruneable_heads_and_indices
            if not hasattr(module, "prune_linear_layer"):
                module.prune_linear_layer = prune_linear_layer

        if not hasattr(clip_modeling, "_expand_mask"):
            def expand_mask(mask, dtype, tgt_len=None):
                batch_size, source_length = mask.shape
                target_length = tgt_len if tgt_len is not None else source_length
                expanded = mask[:, None, None, :].expand(
                    batch_size, 1, target_length, source_length
                ).to(dtype)
                inverted = 1.0 - expanded
                return inverted.masked_fill(inverted.to(torch.bool), torch.finfo(dtype).min)

            clip_modeling._expand_mask = expand_mask

        if "transformers.utils.model_parallel_utils" not in sys.modules:
            parallel_utils = types.ModuleType("transformers.utils.model_parallel_utils")
            parallel_utils.get_device_map = lambda *args, **kwargs: {}
            parallel_utils.assert_device_map = lambda *args, **kwargs: None
            sys.modules[parallel_utils.__name__] = parallel_utils
            transformers_utils.model_parallel_utils = parallel_utils

        from lavis.common.registry import registry
        from lavis.models.base_model import BaseModel
        from lavis.processors.base_processor import BaseProcessor
        from lavis.models.blip2_models.Qformer import BertPreTrainedModel

        # AFPrompt's vendored Q-Former predates Transformers' tied-weight
        # mapping. Transformers 5 expects a mapping when it reports
        # checkpoint-loading diagnostics, so expose the vendored model keys.
        BertPreTrainedModel.all_tied_weights_keys = property(
            lambda model: getattr(model, "_tied_weights_keys", None) or {}
        )

        models_package.BaseModel = BaseModel
        processors_package.BaseProcessor = BaseProcessor
        default_cfg = OmegaConf.load(lavis_root / "configs/default.yaml")
        paths = registry.mapping["paths"]
        for name, value in (
            ("library_root", lavis_root),
            ("repo_root", vendor_root),
            ("cache_root", vendor_root / default_cfg.env.cache_root),
        ):
            if name not in paths:
                registry.register_path(name, str(value))
        from lavis.models.blip2_models.blip2_vicuna_instruct_textinv import (
            Blip2VicunaInstructTextinv,
        )
        from lavis.processors.blip_processors import BlipImageEvalProcessorTextInversion

        vicuna_path = os.environ.get("FAKEPARTS_VICUNA7B_DIR")
        if not vicuna_path and len(self.ckpt_paths) > 2:
            vicuna_path = str(self.ckpt_paths[2])
        if not vicuna_path or not Path(vicuna_path).is_dir():
            raise FileNotFoundError(
                "AntifakePrompt requires local Vicuna v1.3 7B weights; set "
                "FAKEPARTS_VICUNA7B_DIR to that model directory"
            )
        self.torch = torch
        self.device_obj = torch.device(device)
        model_cls = Blip2VicunaInstructTextinv
        cfg = OmegaConf.load(model_cls.default_config_path("vicuna7b"))
        cfg.model.finetuned = str(self.ckpt_path)
        cfg.model.pretrained = str(self.ckpt_paths[1])
        cfg.model.llm_model = str(Path(vicuna_path).expanduser().resolve())
        self.model = model_cls.from_config(cfg.model).to(self.device_obj).eval()
        autocast_dtype = (
            torch.bfloat16 if self.device_obj.type == "cuda"
            and torch.cuda.is_bf16_supported() else torch.float16
        )
        model_autocast = self.model.maybe_autocast
        self.model.maybe_autocast = lambda dtype=None: model_autocast(
            dtype=autocast_dtype if dtype is None else dtype
        )
        if self.device_obj.type == "cpu":
            self.model.float()
        self.image_processor = BlipImageEvalProcessorTextInversion.from_config(
            cfg.preprocess.vis_processor.eval
        )
        self.vendor_root = vendor_root

    def predict_frames(self, img_list: Sequence) -> list[dict]:
        if not img_list:
            return []
        images = [self.image_processor(image.convert("RGB")) for image in img_list]
        batch = self.torch.stack(images).to(self.device_obj)
        samples = {
            "image": batch,
            "prompt": [self.question] * len(images),
        }
        with self.torch.inference_mode():
            # The released evaluator consumes candidate ranks from
            # predict_class(); raw per-token loss values can be non-finite on
            # this checkpoint/runtime combination. Keep the author's decision
            # rule and record its binary real/fake output without presenting it
            # as a calibrated probability.
            candidate_ranks = self.model.predict_class(
                samples=samples, candidates=list(self.candidates),
            )
            top_candidate = candidate_ranks[:, 0].reshape(-1).long()
            scores = (top_candidate == 1).float().cpu().tolist()
        if len(scores) != len(img_list):
            raise ValueError(f"AntifakePrompt returned {len(scores)} scores for {len(img_list)} images")
        return [self.result(score) for score in scores]
