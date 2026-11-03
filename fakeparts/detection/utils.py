from __future__ import annotations

import hashlib
import fcntl
import json
import os
import shlex
import shutil
import subprocess
import sys
from dataclasses import asdict, dataclass
from importlib import import_module
from pathlib import Path, PurePosixPath
from typing import Callable, Iterator
from urllib.error import HTTPError
from urllib.parse import urlencode
from urllib.request import Request, urlopen

from .config import Checkpoint, DetectorConfig
from fakeparts.naming import canonical_generator_id, canonical_task_id


VIDEO_EXTENSIONS = {".avi", ".mkv", ".mov", ".mp4", ".webm"}


@dataclass(frozen=True)
class VideoSample:
    sample_id: str
    video: str
    rel_path: str
    class_name: str
    method: str
    label: str
    frame_count: int | None = None

    def row(self) -> dict[str, str]:
        row = asdict(self)
        row["class"] = row.pop("class_name")
        return row


class VideoDataset:
    """Map-style dataset whose source items are always complete videos."""

    def __init__(self, root: str | Path, samples: list[VideoSample]) -> None:
        self.root = Path(root)
        self.samples = samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> tuple[VideoSample, Path]:
        sample = self.samples[index]
        path = Path(sample.rel_path)
        return sample, path if path.is_absolute() else self.root / path


def collate_videos(items: list[tuple[VideoSample, Path]]) -> list[tuple[VideoSample, Path]]:
    """Keep dataclasses and paths intact instead of tensor-collating them."""
    return items


def load_dataset(root: str | Path) -> list[VideoSample]:
    """Discover videos below ``FakePartsBench/{real,fake}``."""
    root = Path(root)
    if not (root / "real").is_dir() or not (root / "fake").is_dir():
        raise ValueError(f"{root} must contain real/ and fake/ directories")
    samples: list[VideoSample] = []
    for label in ("real", "fake"):
        for path in sorted(
            p for p in (root / label).rglob("*")
            if p.is_file() and p.suffix.lower() in VIDEO_EXTENSIONS
        ):
            rel = path.relative_to(root)
            samples.append(VideoSample(
                sample_id=path.stem, video=path.name, rel_path=rel.as_posix(),
                class_name=rel.parts[1] if len(rel.parts) > 2 else "",
                method=rel.parts[2] if len(rel.parts) > 3 else "", label=label,
            ))
    return samples


def _dataset_root_for_jsonl(path: Path) -> Path:
    """Find the FakePartsBench root used to resolve JSONL file_name values."""
    for parent in (path.parent, *path.parent.parents):
        if (parent / "real").is_dir() and (parent / "fake").is_dir():
            return parent
    return path.parent


def _load_jsonl_inputs(source: Path, dataset_root: str | Path | None = None) -> tuple[Path, list[VideoSample]]:
    root = Path(dataset_root).expanduser().resolve() if dataset_root else _dataset_root_for_jsonl(source)
    samples: list[VideoSample] = []
    with source.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"invalid JSONL at {source}:{line_number}: {error}") from error
            if not isinstance(row, dict):
                raise ValueError(f"expected a JSON object at {source}:{line_number}")

            value = row.get("file_name") or row.get("rel_path") or row.get("path")
            if not value:
                raise ValueError(
                    f"missing file_name, rel_path, or path at {source}:{line_number}"
                )
            video_path = Path(str(value)).expanduser()
            if not video_path.is_absolute():
                candidates = [root / video_path]
                # Also accept paths prefixed by the dataset directory name.
                if video_path.parts and video_path.parts[0] == root.name:
                    candidates.insert(0, root.parent / video_path)
                video_path = next((candidate for candidate in candidates if candidate.is_file()), candidates[0])
            video_path = video_path.resolve()
            if video_path.suffix.lower() not in VIDEO_EXTENSIONS:
                raise ValueError(f"unsupported video extension at {source}:{line_number}: {video_path}")
            if not video_path.is_file():
                raise ValueError(f"video does not exist at {source}:{line_number}: {video_path}")

            try:
                rel_path = video_path.relative_to(root).as_posix()
            except ValueError:
                rel_path = str(video_path)
            parts = PurePosixPath(rel_path).parts
            path_label, path_class, path_method = _path_metadata(parts)
            label_value = row.get("label", path_label)
            if label_value in (0, "0", False):
                label = "real"
            elif label_value in (1, "1", True):
                label = "fake"
            else:
                label = str(label_value or "")
            task = canonical_task_id(str(row.get("task") or row.get("class") or row.get("class_name") or path_class))
            method = canonical_generator_id(str(row.get("method") or path_method))
            sample_id = str(row.get("sample_id") or row.get("id") or video_path.stem)
            samples.append(VideoSample(
                sample_id=sample_id, video=video_path.name, rel_path=rel_path,
                class_name=task, method=method, label=label,
                frame_count=_positive_int_or_none(row.get("frame_count")),
            ))
    if not samples:
        raise ValueError(f"no video rows found in JSONL: {source}")
    return root, samples


def _positive_int_or_none(value: object) -> int | None:
    try:
        number = int(value) if value is not None else None
    except (TypeError, ValueError, OverflowError):
        return None
    return number if number is not None and number > 0 else None


def load_inputs(path: str | Path, dataset_root: str | Path | None = None) -> tuple[Path, list[VideoSample]]:
    """Load a video, folder, benchmark root, or JSONL sample list."""
    source = Path(path).expanduser().resolve()
    if not source.exists():
        raise ValueError(f"input does not exist: {source}")
    if source.is_file():
        if source.suffix.lower() in {".jsonl", ".ndjson"}:
            return _load_jsonl_inputs(source, dataset_root)
        if source.suffix.lower() not in VIDEO_EXTENSIONS:
            raise ValueError(f"unsupported video extension: {source.suffix or '(none)'}")
        root = source.parent
        rel_path = source.name
        label, class_name, method = _path_metadata(source.parts)
        return root, [VideoSample(source.stem, source.name, rel_path, class_name, method, label)]

    if not source.is_dir():
        raise ValueError(f"input is not a file or directory: {source}")
    if (source / "real").is_dir() and (source / "fake").is_dir():
        samples = load_dataset(source)
        if samples:
            return source, samples
        raise ValueError(f"no supported videos found under {source}/real or {source}/fake")

    samples: list[VideoSample] = []
    for video in sorted(
        p for p in source.rglob("*")
        if p.is_file() and p.suffix.lower() in VIDEO_EXTENSIONS
    ):
        rel = video.relative_to(source)
        label, class_name, method = _path_metadata(video.parts)
        samples.append(VideoSample(
            sample_id=video.stem, video=video.name, rel_path=rel.as_posix(),
            class_name=class_name, method=method, label=label,
        ))
    if not samples:
        raise ValueError(f"no supported videos found under {source}")
    return source, samples


def _path_metadata(parts: tuple[str, ...]) -> tuple[str, str, str]:
    """Infer benchmark label/task/method from a FakePartsBench path when present."""
    lowered = tuple(part.lower() for part in parts)
    for label in ("fake", "real"):
        if label not in lowered:
            continue
        index = lowered.index(label)
        tail = parts[index + 1:]
        class_name = tail[0] if len(tail) > 1 else ""
        method = tail[1] if len(tail) > 2 else ""
        return label, class_name, method
    return "", "", ""


def _frame_cache_key(video_path: Path) -> str:
    stat = video_path.stat()
    identity = f"{video_path.resolve()}\0{stat.st_size}\0{stat.st_mtime_ns}".encode()
    return hashlib.sha256(identity).hexdigest()


def cache_video_frames(video_path: str | Path, cache_root: str | Path) -> Path:
    """Decode a video once into a reusable, run-scoped JPEG frame cache."""
    video_path = Path(video_path).expanduser().resolve()
    cache_root = Path(cache_root).expanduser().resolve()
    cache_dir = cache_root / _frame_cache_key(video_path)
    lock_dir = cache_root / ".locks"
    lock_dir.mkdir(parents=True, exist_ok=True)
    lock_path = lock_dir / f"{cache_dir.name}.lock"
    with lock_path.open("a+") as lock_file:
        fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
        try:
            return _cache_video_frames_locked(video_path, cache_dir)
        finally:
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)


def _cache_video_frames_locked(video_path: Path, cache_dir: Path) -> Path:
    dependency_root = Path(__file__).resolve().parents[2] / ".deps" / "detection-extra"
    added_dependency_root = dependency_root.is_dir() and str(dependency_root) not in sys.path
    if added_dependency_root:
        # Keep the optional dependency directory at the end: it can contain
        # method-specific torch wheels, while frame extraction only needs cv2.
        sys.path.append(str(dependency_root))
    try:
        import cv2
    finally:
        if added_dependency_root:
            sys.path.remove(str(dependency_root))

    metadata_path = cache_dir / "frames.json"
    if metadata_path.is_file():
        try:
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            count = int(metadata["frame_count"])
            if (metadata.get("source") == str(video_path)
                    and count > 0
                    and len(list(cache_dir.glob("frame_*.jpg"))) == count):
                return cache_dir
        except (OSError, ValueError, KeyError, TypeError, json.JSONDecodeError):
            pass

    cache_dir.mkdir(parents=True, exist_ok=True)
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise RuntimeError(f"cannot open video: {video_path}")
    index = 0
    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            output_path = cache_dir / f"frame_{index:08d}.jpg"
            if not cv2.imwrite(str(output_path), frame):
                raise RuntimeError(f"failed to cache frame {index}: {output_path}")
            index += 1
        if index == 0:
            raise RuntimeError(f"no decodable frames: {video_path}")
    finally:
        capture.release()
    metadata_tmp = cache_dir / "frames.json.tmp"
    metadata_tmp.write_text(json.dumps({
        "source": str(video_path),
        "source_size": video_path.stat().st_size,
        "source_mtime_ns": video_path.stat().st_mtime_ns,
        "frame_count": index,
    }, indent=2) + "\n", encoding="utf-8")
    metadata_tmp.replace(metadata_path)
    return cache_dir


def iter_frame_batches(
    video_path: str | Path,
    batch_size: int | Callable[[], int],
    cache_dir: str | Path | None = None,
) -> Iterator[tuple[list[int], list[object]]]:
    """Read RGB PIL batches from a reusable frame cache or decode the source video."""
    from PIL import Image

    def current_batch_size() -> int:
        value = batch_size() if callable(batch_size) else batch_size
        if value < 1:
            raise ValueError(f"frame batch size must be positive, got {value}")
        return value

    if cache_dir is not None:
        frame_dir = cache_video_frames(video_path, cache_dir)
        frame_paths = sorted(frame_dir.glob("frame_*.jpg"))
        start = 0
        while start < len(frame_paths):
            paths = frame_paths[start:start + current_batch_size()]
            images = []
            for path in paths:
                with Image.open(path) as image:
                    images.append(image.convert("RGB"))
            yield list(range(start, start + len(paths))), images
            start += len(paths)
        return

    import cv2

    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise RuntimeError(f"cannot open video: {video_path}")
    indices: list[int] = []
    images: list[object] = []
    try:
        index = 0
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            indices.append(index)
            images.append(Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)))
            index += 1
            if len(images) == current_batch_size():
                yield indices, images
                indices, images = [], []
        if images:
            yield indices, images
        if index == 0:
            raise RuntimeError(f"no decodable frames: {video_path}")
    finally:
        capture.release()


def ensure_checkpoints(
    checkpoints: tuple[Checkpoint, ...],
    download: bool = False,
    force_download: bool = False,
) -> tuple[Path, ...]:
    """Check configured checkpoint files and optionally fetch or refresh them."""
    paths: list[Path] = []
    missing: list[Path] = []
    for checkpoint in checkpoints:
        path = checkpoint.path
        valid = _checkpoint_is_present(checkpoint)
        should_download = force_download or not valid
        if should_download and (download or force_download):
            hf_error: Exception | None = None
            if checkpoint.hf_repo:
                try:
                    _download_huggingface_checkpoint(
                        checkpoint, force_download=force_download
                    )
                except (RuntimeError, OSError) as error:
                    hf_error = error
                    if not checkpoint.gdrive_id and not checkpoint.url:
                        raise
                    print(
                        f"Hugging Face source failed for {path}; trying the "
                        "registered fallback source.", file=sys.stderr, flush=True,
                    )
            if not _checkpoint_is_present(checkpoint) and checkpoint.gdrive_id:
                _download_google_drive_checkpoint(checkpoint)
            elif not _checkpoint_is_present(checkpoint) and checkpoint.url:
                path.parent.mkdir(parents=True, exist_ok=True)
                temporary = path.with_suffix(path.suffix + ".part")
                try:
                    with urlopen(checkpoint.url) as source, temporary.open("wb") as target:
                        while chunk := source.read(1024 * 1024):
                            target.write(chunk)
                    temporary.replace(path)
                except Exception:
                    if hf_error:
                        raise RuntimeError(
                            f"Hugging Face and fallback downloads both failed for {path}"
                        ) from hf_error
                    raise
                finally:
                    temporary.unlink(missing_ok=True)
            elif not _checkpoint_is_present(checkpoint) and not checkpoint.gdrive_id and not checkpoint.url:
                # Leave source-less local checkpoints to the missing-file report.
                pass
        valid = _checkpoint_is_present(checkpoint)
        if not valid:
            missing.append(path)
        paths.append(path)
    if missing:
        formatted = "\n".join(f"  - {path}" for path in missing)
        raise FileNotFoundError(f"missing checkpoints:\n{formatted}")
    return tuple(paths)


def _checkpoint_is_present(checkpoint: Checkpoint) -> bool:
    path = checkpoint.path
    return path.is_file() or (
        path.is_dir() and bool(checkpoint.required_files)
        and all((path / name).is_file() for name in checkpoint.required_files)
    )


def _download_huggingface_checkpoint(
    checkpoint: Checkpoint, *, force_download: bool = False
) -> None:
    """Fetch a Hub file or snapshot through the Hugging Face Python API."""
    path = checkpoint.path
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        from huggingface_hub import hf_hub_download, snapshot_download
    except ImportError as error:
        raise RuntimeError(
            "Install the project environment with `uv sync` to use "
            "huggingface_hub for checkpoint downloads."
        ) from error

    try:
        if checkpoint.hf_filename:
            filename_parts = PurePosixPath(checkpoint.hf_filename).parts
            local_dir = path.parents[len(filename_parts) - 1]
            downloaded = Path(hf_hub_download(
                repo_id=checkpoint.hf_repo,
                filename=checkpoint.hf_filename,
                local_dir=local_dir,
                force_download=force_download,
            ))
            if downloaded.resolve() != path.resolve():
                raise RuntimeError(
                    f"Hub path mismatch: expected {path}, downloaded {downloaded}"
                )
        else:
            snapshot_download(
                repo_id=checkpoint.hf_repo,
                local_dir=path,
                force_download=force_download,
            )
    except Exception as error:
        raise RuntimeError(
            f"Hugging Face download failed for {checkpoint.hf_repo}"
            + (f"/{checkpoint.hf_filename}" if checkpoint.hf_filename else "")
        ) from error


def _gdown_prefix() -> list[str] | None:
    executable = shutil.which("gdown")
    if executable:
        return [executable]
    uvx = shutil.which("uvx")
    if uvx:
        return [uvx, "--from", "gdown", "gdown"]
    return None


def _google_drive_api_download(file_id: str, destination: Path, api_key: str) -> None:
    query = urlencode({"alt": "media", "key": api_key})
    request = Request(f"https://www.googleapis.com/drive/v3/files/{file_id}?{query}")
    try:
        with urlopen(request, timeout=60) as response:
            content_type = response.headers.get("Content-Type", "")
            if "json" in content_type.lower():
                detail = response.read(4096).decode("utf-8", errors="replace")
                raise RuntimeError(f"Google Drive API returned an error: {detail}")
            with destination.open("wb") as output:
                shutil.copyfileobj(response, output, length=1024 * 1024)
    except HTTPError as error:
        detail = error.read(4096).decode("utf-8", errors="replace")
        raise RuntimeError(f"Google Drive API HTTP {error.code}: {detail}") from error


def _google_drive_api_find_file(folder_id: str, filename: str, api_key: str) -> str:
    wanted = PurePosixPath(filename).as_posix().lstrip("./")
    queue = [(folder_id, "")]
    matches: list[str] = []
    while queue:
        parent_id, prefix = queue.pop(0)
        page_token = None
        while True:
            query = f"'{parent_id}' in parents and trashed = false"
            params = {
                "q": query,
                "fields": "nextPageToken,files(id,name,mimeType)",
                "pageSize": "1000",
                "key": api_key,
            }
            if page_token:
                params["pageToken"] = page_token
            request = Request("https://www.googleapis.com/drive/v3/files?" + urlencode(params))
            with urlopen(request, timeout=60) as response:
                result = json.load(response)
            for item in result.get("files", []):
                rel = f"{prefix}/{item['name']}" if prefix else item["name"]
                if item.get("mimeType") == "application/vnd.google-apps.folder":
                    queue.append((item["id"], rel))
                elif rel == wanted or ("/" not in wanted and item["name"] == wanted):
                    matches.append(item["id"])
            page_token = result.get("nextPageToken")
            if not page_token:
                break
    if len(matches) != 1:
        raise RuntimeError(f"Drive API found {len(matches)} matches for {filename!r} in folder {folder_id}")
    return matches[0]


def _download_google_drive_checkpoint(checkpoint: Checkpoint) -> None:
    path = checkpoint.path
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".download")
    file_url = f"https://drive.google.com/uc?id={checkpoint.gdrive_id}"
    cli = _gdown_prefix()
    errors: list[str] = []

    if cli:
        if checkpoint.gdrive_filename:
            folder_url = f"https://drive.google.com/drive/folders/{checkpoint.gdrive_id}"
            listing = subprocess.run(
                [*cli, folder_url, "--json", "--quiet"],
                check=False, capture_output=True, text=True,
            )
            if listing.returncode == 0:
                try:
                    items = json.loads(listing.stdout)
                    wanted = PurePosixPath(checkpoint.gdrive_filename).as_posix().lstrip("./")
                    exact = [item for item in items if item.get("path") == wanted]
                    candidates = exact or [
                        item for item in items
                        if PurePosixPath(item.get("path", "")).name == PurePosixPath(wanted).name
                    ]
                    if len(candidates) == 1:
                        file_url = candidates[0]["url"]
                    else:
                        errors.append(
                            f"gdown folder listing found {len(candidates)} matches for "
                            f"{checkpoint.gdrive_filename!r}"
                        )
                except (json.JSONDecodeError, KeyError, TypeError) as error:
                    errors.append(f"could not parse gdown folder listing: {error}")
            else:
                errors.append(f"gdown folder listing exited with status {listing.returncode}: {listing.stderr.strip()}")

        if not errors or file_url != f"https://drive.google.com/uc?id={checkpoint.gdrive_id}":
            command = [*cli, "--continue", "--retries", "3", file_url, "-O", str(temporary)]
            print(f"[checkpoint] Downloading Google Drive file {checkpoint.gdrive_id} -> {path}", flush=True)
            result = subprocess.run(command, check=False)
            if result.returncode == 0 and temporary.is_file() and temporary.stat().st_size > 0:
                temporary.replace(path)
                return
            errors.append(f"gdown exited with status {result.returncode}")
    else:
        errors.append("neither gdown nor uvx is installed")

    api_key = os.getenv("FAKEPARTS_GOOGLE_DRIVE_API_KEY") or os.getenv("GOOGLE_DRIVE_API_KEY")
    if api_key:
        try:
            file_id = checkpoint.gdrive_id
            if checkpoint.gdrive_filename:
                file_id = _google_drive_api_find_file(file_id, checkpoint.gdrive_filename, api_key)
            _google_drive_api_download(file_id, temporary, api_key)
            if temporary.is_file() and temporary.stat().st_size > 0:
                temporary.replace(path)
                return
            errors.append("Google Drive API returned an empty file")
        except Exception as error:
            errors.append(str(error))

    manual_url = (f"https://drive.google.com/drive/folders/{checkpoint.gdrive_id}"
                  if checkpoint.gdrive_filename else file_url)
    folder_flag = " --folder" if checkpoint.gdrive_filename else ""
    manual = (f"uvx --from gdown gdown{folder_flag} --continue --retries 3 "
              f"{shlex.quote(manual_url)} -O {shlex.quote(str(path.parent)) if checkpoint.gdrive_filename else shlex.quote(str(path))}")
    suggestions = [
        f"Automatic Google Drive download failed for checkpoint: {path}",
        *[f"  {error}" for error in errors],
        f"Manual download command: {manual}",
        "For public files/folders, set FAKEPARTS_GOOGLE_DRIVE_API_KEY (or GOOGLE_DRIVE_API_KEY) to enable the Drive API fallback.",
        "If the file is restricted, grant access to the account or download it manually to the checkpoint path above.",
        f"Share URL: {manual_url}",
    ]
    raise RuntimeError("\n".join(suggestions))


def load_detector(config: DetectorConfig, checkpoints: tuple[Path, ...], **kwargs):
    """Import and construct an adapter with all of its checked checkpoints."""
    if not checkpoints:
        raise ValueError(f"no checkpoints configured for {config.class_name}")
    project_root = Path(__file__).resolve().parents[2]
    for dependency_root in (
        project_root / ".deps" / "detection-extra",
        project_root / ".deps" / "detection",
    ):
        if dependency_root.is_dir() and str(dependency_root) not in sys.path:
            sys.path.append(str(dependency_root))
    module = import_module(
        f"fakeparts.detection.{config.family}.{config.module}.detector")
    detector_class = getattr(module, config.adapter_class or config.class_name)
    return detector_class(ckpt_path=checkpoints[0], ckpt_paths=checkpoints, **kwargs)
