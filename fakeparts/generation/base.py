"""Shared generation runner used by the model-specific adapters."""

from __future__ import annotations

from dataclasses import dataclass
import json
import shlex
import subprocess
from pathlib import Path
from typing import Iterable, Sequence


@dataclass(frozen=True)
class GenerationJob:
    sample_id: str
    command: tuple[str, ...]
    inputs: tuple[str, ...]


class Generator:
    """Run validated adapter commands one sample at a time.

    Model weights and dependencies stay in each adapter's environment. This
    class only handles the common process lifecycle and structured logging.
    """

    def __init__(
        self, *, dry_run: bool = False, continue_on_error: bool = False,
        setup_method: str | None = None, setup_available: bool = False,
    ) -> None:
        self.dry_run = dry_run
        self.continue_on_error = continue_on_error
        self.setup_method = setup_method
        self.setup_available = setup_available

    def run_one(self, job: GenerationJob) -> int:
        if self.dry_run:
            print(json.dumps({
                "sample_id": job.sample_id,
                "inputs": list(job.inputs),
                "command": list(job.command),
            }, ensure_ascii=False))
            return 0
        print(f"[generate] {job.sample_id}", flush=True)
        process = subprocess.Popen(
            job.command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True, bufsize=1,
        )
        missing_weight_hint = False
        missing_markers = ("no such file", "not found", "does not exist", "missing", "cannot load")
        weight_terms = ("checkpoint", "weight", "pretrained", ".pth", ".ckpt", ".safetensors", ".onnx")
        assert process.stdout is not None
        for line in process.stdout:
            print(line, end="", flush=True)
            lowered = line.lower()
            if any(marker in lowered for marker in missing_markers) and any(term in lowered for term in weight_terms):
                missing_weight_hint = True
        returncode = process.wait()
        if returncode:
            print(f"[generate] failed ({returncode}): {job.sample_id}", flush=True)
            if missing_weight_hint:
                self._print_setup_hint()
        return returncode

    def _print_setup_hint(self) -> None:
        print("[generate] The error looks like a missing model weight.", flush=True)
        if not self.setup_method or not self.setup_available:
            print("[generate] No automated setup profile is registered for this method; see its method README for weight setup.", flush=True)
            return
        target = shlex.quote(self.setup_method)
        setup_script = Path(__file__).resolve().parents[2] / "scripts" / "set_up_generator.sh"
        script = shlex.quote(str(setup_script))
        print(f"[generate] Set up weights with: bash {script} --methods {target}", flush=True)
        print(f"[generate] Or submit it with: sbatch {script} --methods {target}", flush=True)

    def run(self, jobs: Iterable[GenerationJob]) -> int:
        failures = 0
        total = 0
        for job in jobs:
            total += 1
            code = self.run_one(job)
            if code:
                failures += 1
                if not self.continue_on_error:
                    break
        print(f"[generate] finished: {total - failures}/{total} succeeded", flush=True)
        return 1 if failures else 0


def command_for(entrypoint: str, python: str, arguments: Sequence[str]) -> tuple[str, ...]:
    return (python, entrypoint, *arguments)
