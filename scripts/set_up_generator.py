#!/usr/bin/env python3
"""Prepare explicitly selected generator environments and checkpoints."""
from __future__ import annotations
import argparse
import hashlib
import io
import os
import subprocess
import sys
import urllib.error
import urllib.request
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from fakeparts.generation.config import MODEL_SPECS

SETUP_PROFILES = {"insightface"}

GENERATOR_METHOD_GROUPS = {
    "Face swap": ("hunyuanportrait", "insightface", "liveportrait"),
    "T2V": (
        "allegroai", "cogvidx", "hunyuanvid", "ltx_video", "mochi1",
        "opensora:t2v", "wan21:t2v",
    ),
    "IT2V": ("opensora:it2v", "wan21:it2v"),
    "Extrapolation": ("cosmos2",),
    "Interpolation": ("framer",),
    "Inpainting": ("diffueraser", "propainter", "rovi"),
    "Outpainting": ("akira",),
    "Style transfer": ("anyv2v", "rave"),
}


def generator_methods_help() -> str:
    return (
        "Methods accepted by this setup script: "
        f"{', '.join(sorted(SETUP_PROFILES))}. Registered generator methods by task:\n"
        + "\n".join(
            f"  {label}: {', '.join(methods)}" for label, methods in GENERATOR_METHOD_GROUPS.items()
        )
        + "\n  Defaults to all setup methods."
    )

INSIGHTFACE_ROOT = ROOT / "ckpt/generation/InsightFace"
LEGACY_INSIGHTFACE_ROOT = ROOT.parent / "FakePartsBench/ckpt/generation/InsightFace"
BUFFALO_L_URL = "https://github.com/deepinsight/insightface/releases/download/model-zoo/buffalo_l.zip"
INSWAPPER_URL = "https://github.com/deepinsight/insightface/releases/download/model-zoo/inswapper_128.onnx"
INSIGHTFACE_FILES = {
    "inswapper_128.onnx": (554253681, "e4a3f08c753cb72d04e10aa0f7dbe3deebbf39567d4ead6dce08e98aa49e16af"),
    "models/buffalo_l/1k3d68.onnx": (143607619, "df5c06b8a0c12e422b2ed8947b8869faa4105387f199c477af038aa01f9a45cc"),
    "models/buffalo_l/2d106det.onnx": (5030888, "f001b856447c413801ef5c42091ed0cd516fcd21f2d6b79635b1e733a7109dbf"),
    "models/buffalo_l/det_10g.onnx": (16923827, "5838f7fe053675b1c7a08b633df49e7af5495cee0493c7dcf6697200b85b5b91"),
    "models/buffalo_l/genderage.onnx": (1322532, "4fde69b1c810857b88c64a335084f1c3fe8f01246c9a191b48c7bb756d6652fb"),
    "models/buffalo_l/w600k_r50.onnx": (174383860, "4c06341c33c2fa1f86781dab0e829f88ad5b64be9fba56e56bc9ebdefc619e43"),
}


def file_status(path: Path, expected: tuple[int, str]) -> str:
    expected_size, expected_sha256 = expected
    if not path.is_file():
        return "MISSING"
    if path.stat().st_size != expected_size:
        return "SIZE_MISMATCH"
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return "OK" if digest.hexdigest() == expected_sha256 else "HASH_MISMATCH"


def report_insightface_status(root: Path, *, prefix: str = "insightface") -> bool:
    complete = True
    for relative, expected in INSIGHTFACE_FILES.items():
        status = file_status(root / relative, expected)
        print(f"  {prefix} {relative} STATUS {status}")
        complete &= status == "OK"
    return complete


def download_bytes(url: str, *, label: str, expected_size: int | None = None) -> bytes:
    request = urllib.request.Request(url, headers={"User-Agent": "FakeParts-generator-setup/1.0"})
    try:
        with urllib.request.urlopen(request, timeout=60) as response:
            total = int(response.headers.get("Content-Length", "0")) or expected_size or 0
            chunks: list[bytes] = []
            received = 0
            while True:
                chunk = response.read(8 * 1024 * 1024)
                if not chunk:
                    break
                chunks.append(chunk)
                received += len(chunk)
                if total:
                    print(f"  {label}: {received / 1024**2:.0f}/{total / 1024**2:.0f} MiB", end="\r", flush=True)
            print(f"  {label}: downloaded {received / 1024**2:.1f} MiB")
            return b"".join(chunks)
    except (urllib.error.URLError, TimeoutError) as exc:
        raise RuntimeError(
            f"Could not download {label} from {url}: {exc}. "
            f"You can set FAKEPARTS_{'BUFFALO_L' if 'buffalo_l' in label else 'INSWAPPER'}_URL to a reachable mirror."
        ) from exc


def install_inswapper(*, overwrite: bool) -> None:
    relative = "inswapper_128.onnx"
    destination = INSIGHTFACE_ROOT / relative
    expected = INSIGHTFACE_FILES[relative]
    status = file_status(destination, expected)
    if status == "OK" and not overwrite:
        return
    url = os.getenv("FAKEPARTS_INSWAPPER_URL", INSWAPPER_URL)
    payload = download_bytes(url, label=relative, expected_size=expected[0])
    digest = hashlib.sha256(payload).hexdigest()
    if len(payload) != expected[0] or digest != expected[1]:
        raise RuntimeError(
            f"Downloaded {relative} failed integrity validation (size={len(payload)}, sha256={digest}). "
            "Check the URL or the upstream model version."
        )
    destination.parent.mkdir(parents=True, exist_ok=True)
    staged = destination.with_name(destination.name + ".download")
    staged.write_bytes(payload)
    os.replace(staged, destination)
    print(f"  Installed {destination}")


def install_buffalo_l(*, overwrite: bool) -> None:
    missing = [
        (relative, expected)
        for relative, expected in INSIGHTFACE_FILES.items()
        if relative.startswith("models/buffalo_l/")
        and (overwrite or file_status(INSIGHTFACE_ROOT / relative, expected) != "OK")
    ]
    if not missing:
        return
    url = os.getenv("FAKEPARTS_BUFFALO_L_URL", BUFFALO_L_URL)
    payload = download_bytes(url, label="buffalo_l.zip")
    try:
        archive = zipfile.ZipFile(io.BytesIO(payload))
    except zipfile.BadZipFile as exc:
        raise RuntimeError(
            f"Downloaded buffalo_l archive is invalid. Check FAKEPARTS_BUFFALO_L_URL={url}"
        ) from exc
    by_basename: dict[str, list[str]] = {}
    for name in archive.namelist():
        if not name.endswith("/"):
            by_basename.setdefault(Path(name).name, []).append(name)
    for relative, expected in missing:
        basename = Path(relative).name
        candidates = by_basename.get(basename, [])
        if len(candidates) != 1:
            raise RuntimeError(f"Expected one {basename} in buffalo_l.zip; found {len(candidates)}")
        data = archive.read(candidates[0])
        digest = hashlib.sha256(data).hexdigest()
        if len(data) != expected[0] or digest != expected[1]:
            raise RuntimeError(f"{basename} from buffalo_l.zip failed size/SHA-256 validation")
        destination = INSIGHTFACE_ROOT / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        staged = destination.with_name(destination.name + ".download")
        staged.write_bytes(data)
        os.replace(staged, destination)
        print(f"  Installed {destination}")


def setup_insightface(*, overwrite: bool) -> int:
    canonical_complete = all(
        file_status(INSIGHTFACE_ROOT / relative, expected) == "OK"
        for relative, expected in INSIGHTFACE_FILES.items()
    )
    legacy_complete = all(
        file_status(LEGACY_INSIGHTFACE_ROOT / relative, expected) == "OK"
        for relative, expected in INSIGHTFACE_FILES.items()
    )
    if legacy_complete and not canonical_complete and not overwrite:
        print(f"InsightFace weights already complete in legacy location: {LEGACY_INSIGHTFACE_ROOT}")
        return 0
    try:
        setup_root = INSIGHTFACE_ROOT
        setup_root.mkdir(parents=True, exist_ok=True)
        install_inswapper(overwrite=overwrite)
        install_buffalo_l(overwrite=overwrite)
    except (OSError, RuntimeError) as exc:
        print(f"ERROR: InsightFace setup failed: {exc}", file=sys.stderr)
        return 1
    return 0

def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument("--methods", action="extend", nargs="+", default=None, metavar="METHOD",
                        help=generator_methods_help())
    parser.add_argument("--dry-run", action="store_true", help="show selected setup without executing it")
    parser.add_argument("--overwrite", action="store_true", help="force-refresh selected model weights")
    args = parser.parse_args()
    if args.methods is None:
        args.methods = sorted(SETUP_PROFILES)
        print("--methods omitted; selecting all configured generator setups: " + ", ".join(args.methods))
    selected = []
    for token in args.methods:
        for raw in token.split(","):
            name = raw.strip().lower().replace("-", "_")
            if name not in MODEL_SPECS:
                parser.error(f"unknown generator {raw!r}; registered methods: {', '.join(sorted(MODEL_SPECS))}")
            if name not in SETUP_PROFILES:
                parser.error(f"{name} has no complete shared setup profile yet; setup skipped")
            if name not in selected:
                selected.append(name)
    if not selected:
        parser.error("--methods must contain at least one method ID")
    print("Selected generator setup: " + ", ".join(selected))
    if args.dry_run:
        for name in selected:
            if name == "insightface":
                canonical_ok = report_insightface_status(INSIGHTFACE_ROOT)
                legacy_ok = report_insightface_status(LEGACY_INSIGHTFACE_ROOT, prefix="insightface-legacy")
                if not canonical_ok and legacy_ok:
                    print(f"  insightface will reuse legacy weights from {LEGACY_INSIGHTFACE_ROOT}")
                else:
                    print(f"  insightface download target={INSIGHTFACE_ROOT}")
            else:
                env_name = ".venv-hunyuanportrait" if name == "hunyuanportrait" else ".venv-liveportrait"
                model_dir = ROOT / "fakeparts/generation/faceswap" / name
                print(f"  {name}: environment={ROOT / env_name}")
                print(f"  weights={model_dir / 'pretrained_weights'} (download missing/outdated files)")
        if args.overwrite:
            print("  overwrite: force-refresh configured weights")
        return 0
    status = 0
    if "insightface" in selected:
        status = setup_insightface(overwrite=args.overwrite)
        if status:
            return status
    portrait_methods = [name for name in selected if name in {"hunyuanportrait", "liveportrait"}]
    if portrait_methods:
        setup_script = ROOT / "scripts/portrait_setup.sh"
        if not setup_script.is_file():
            print(
                "ERROR: portrait setup script is absent; InsightFace setup (if selected) is complete, "
                f"but cannot prepare {', '.join(portrait_methods)}.",
                file=sys.stderr,
            )
            return 2
        env = os.environ.copy()
        env["FAKEPARTS_PORTRAIT_METHODS"] = ",".join(portrait_methods)
        env["FAKEPARTS_SETUP_OVERWRITE"] = "1" if args.overwrite else "0"
        return subprocess.run(["bash", str(setup_script)], cwd=ROOT, env=env).returncode
    return status

if __name__ == "__main__":
    raise SystemExit(main())
