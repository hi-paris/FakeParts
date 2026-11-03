from __future__ import annotations

import os
import re
from pathlib import Path

from ...base import VideoDetector


class BusterX(VideoDetector):
    """BusterX++ video question-answering adapter using its released prompt."""

    prompt = (
        "Please analyze whether there are any inconsistencies or obvious signs of forgery in the video, "
        "and finally come to a conclusion: Is this video real or fake?\n\n"
        "Then, just answer this MCQ with a single letter:\n"
        "Q: Is this video real or fake?\n"
        "Options:\nA) real\nB) fake\n\n"
        "Put the final answer in \\\\boxed{...}"
    )
    answer_pattern = re.compile(r"\\boxed\s*\{\s*([AB])\s*\}", re.IGNORECASE)

    def __init__(self, ckpt_path, device="cuda", threshold=0.5, crop_size=None,
                 work_dir=None, ckpt_paths=None, version="default"):
        super().__init__(ckpt_path, device, threshold, crop_size, work_dir, ckpt_paths, version)
        import torch
        from transformers import AutoProcessor, Qwen3_5ForConditionalGeneration

        if not Path(self.ckpt_path).is_dir():
            raise FileNotFoundError(f"BusterX expects a local Hugging Face model directory: {self.ckpt_path}")
        self.torch = torch
        self.processor = AutoProcessor.from_pretrained(self.ckpt_path, local_files_only=True)
        dtype = torch.bfloat16 if str(device).startswith("cuda") else torch.float32
        self.max_new_tokens = max(1, int(os.getenv("FAKEPARTS_BUSTERX_MAX_NEW_TOKENS", "4096")))
        self.model = Qwen3_5ForConditionalGeneration.from_pretrained(
            self.ckpt_path, torch_dtype=dtype, low_cpu_mem_usage=True,
            local_files_only=True,
        ).to(device).eval()
        self.device = device

    def predict_video(self, video_path):
        path = Path(video_path).expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(path)
        messages = [{
            "role": "user",
            "content": [{"type": "video", "url": str(path)}, {"type": "text", "text": self.prompt}],
        }]
        inputs = self.processor.apply_chat_template(
            messages, tokenize=True, add_generation_prompt=True, return_dict=True,
            return_tensors="pt", video_fps=1.0,
        )
        inputs = {
            key: value.to(self.device) if hasattr(value, "to") else value
            for key, value in inputs.items()
        }
        with self.torch.inference_mode():
            # The authors' BusterX++ inference scripts default to 4096 generated
            # tokens. Qwen may use a long reasoning trace before its boxed
            # verdict, so a shorter cap leaves otherwise valid samples without
            # the answer expected by their evaluator.
            output_ids = self.model.generate(
                **inputs, do_sample=False, max_new_tokens=self.max_new_tokens
            )
        response_ids = output_ids[0, inputs["input_ids"].shape[1]:]
        response = self.processor.decode(response_ids, skip_special_tokens=True)
        match = self.answer_pattern.search(response)
        if match is None:
            raise ValueError(f"BusterX response has no boxed A/B answer: {response[-300:]!r}")
        # The official evaluator judges a boxed answer; it does not expose calibrated
        # probabilities. Preserve that categorical prediction in the runner's score.
        fake_score = float(match.group(1).upper() == "B")
        return self.result(fake_score)
