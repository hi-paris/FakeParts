from __future__ import annotations

from pathlib import Path

from ...base import FrameDetector


class FakeVLM(FrameDetector):
    """FakeVLM LLaVA classifier, scored by competing answer token likelihoods."""

    prompt = (
        "<image>\nIs this image real or fake? Consider whether it is authentic or AI-generated. "
        "Answer with exactly one word: real or fake.\nASSISTANT:"
    )

    def __init__(self, ckpt_path, device="cuda", threshold=0.5, crop_size=None,
                 work_dir=None, ckpt_paths=None, version="default"):
        super().__init__(ckpt_path, device, threshold, crop_size, work_dir, ckpt_paths, version)
        import torch
        from transformers import AutoProcessor, LlavaForConditionalGeneration

        if not Path(self.ckpt_path).is_dir():
            raise FileNotFoundError(f"FakeVLM expects a local Hugging Face model directory: {self.ckpt_path}")
        self.torch = torch
        self.processor = AutoProcessor.from_pretrained(self.ckpt_path, local_files_only=True)
        dtype = torch.bfloat16 if str(device).startswith("cuda") else torch.float32
        self.model = LlavaForConditionalGeneration.from_pretrained(
            self.ckpt_path, torch_dtype=dtype, low_cpu_mem_usage=True,
            local_files_only=True,
        ).to(device).eval()
        # The downloaded LLaVA checkpoint's processor metadata omits the
        # vision CLS token, while the model config expects it. Keep the
        # processor's image-token expansion aligned with the actual vision
        # encoder output (patch size and selection strategy are checkpoint
        # specific).
        vision_config = self.model.config.vision_config
        self.processor.patch_size = vision_config.patch_size
        self.processor.vision_feature_select_strategy = (
            self.model.config.vision_feature_select_strategy
        )
        self.processor.num_additional_image_tokens = 1
        tokenizer = self.processor.tokenizer
        self.candidate_ids = {
            label: tokenizer.encode(f" {label}", add_special_tokens=False)
            for label in ("real", "fake")
        }
        if any(not ids for ids in self.candidate_ids.values()):
            raise RuntimeError("FakeVLM tokenizer returned an empty real/fake label")
        self.device = device

    def _candidate_log_probability(self, model_inputs, label_ids):
        torch = self.torch
        prefix_ids = model_inputs["input_ids"]
        prefix_length = prefix_ids.shape[1]
        continuation = torch.tensor([label_ids], dtype=prefix_ids.dtype, device=prefix_ids.device)
        candidate_inputs = dict(model_inputs)
        candidate_inputs["input_ids"] = torch.cat((prefix_ids, continuation), dim=1)
        candidate_inputs["attention_mask"] = torch.cat((
            model_inputs["attention_mask"],
            torch.ones_like(continuation),
        ), dim=1)
        output = self.model(**candidate_inputs)
        token_logits = output.logits[0, prefix_length - 1:prefix_length + len(label_ids) - 1]
        token_log_probs = token_logits.log_softmax(dim=-1)
        target = continuation[0]
        return token_log_probs.gather(1, target[:, None]).sum()

    def predict_frames(self, images):
        torch = self.torch
        results = []
        for image in images:
            inputs = self.processor(
                text=self.prompt, images=image.convert("RGB"), return_tensors="pt",
            )
            model_inputs = {
                key: value.to(self.device) if hasattr(value, "to") else value
                for key, value in inputs.items()
            }
            with torch.inference_mode():
                real_logp = self._candidate_log_probability(
                    model_inputs, self.candidate_ids["real"]
                )
                fake_logp = self._candidate_log_probability(
                    model_inputs, self.candidate_ids["fake"]
                )
                fake_score = torch.stack((real_logp, fake_logp)).softmax(0)[1].item()
            results.append(self.result(fake_score))
        return results
