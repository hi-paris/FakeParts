# InsightFace Python Library 2.0

InsightFace provides face detection, recognition, alignment, and attributes
through Python and ONNX Runtime. Optional features include RGB liveness,
PrivateFrame for video face blur/mosaic, and the Evaluation Studio desktop GUI.

## License

The library code is released under the **MIT License**, for academic and
commercial use. **The pretrained models provided with this library are for
non-commercial research only**, whether downloaded automatically or manually.

## What's new in 2.0

- **[Liveness update](#optional-liveness-addon):** optional RGB liveness before
  recognition, configurable recognition gating, and per-face results.
- **[PrivateFrame update](#privateframe):** local video face blur/mosaic,
  reference-photo selection, editable analysis JSON, and desktop, CLI, and
  Python API workflows. See the [full guide](https://github.com/deepinsight/insightface/blob/master/python-package/insightface/app/privateframe/README.md).
- **Runtime and models:** `raccoon_s` / `raccoon_l`, automatic CoreML/CUDA/CPU
  selection, and reusable CoreML compilation caches.

## Installation

Python 3.10 or newer is required.

| Use case | Command |
|---|---|
| FaceAnalysis and ModelZoo | `pip install insightface` |
| PrivateFrame API and CLI | `pip install "insightface[privateframe]"` |
| Evaluation Studio GUI, including PrivateFrame | `pip install "insightface[gui]"` |

The base package installs `onnxruntime`. The `privateframe` extra adds PyAV and
PyYAML; the `gui` extra also includes the Qt desktop application. The optional
`face3d` extension is not compiled by default, so ordinary installation does
not require a C++ compiler. See the
[source installation and runtime guide](https://github.com/deepinsight/insightface/blob/master/python-package/docs/runtime.md)
for installation details.

### NVIDIA CUDA

After installing InsightFace, replace the default runtime with the GPU
distribution:

```bash
python -m pip uninstall -y onnxruntime
python -m pip install onnxruntime-gpu
```

Do not keep both runtime distributions installed together. Installing or
upgrading InsightFace may install `onnxruntime` again; repeat this replacement
afterward on NVIDIA systems.

## Quick Example

Detect faces in the bundled sample image and save an annotated image:

```python
import cv2
from insightface.app import FaceAnalysis
from insightface.data import get_image

app = FaceAnalysis()
app.prepare()
image = get_image("t1")
faces = app.get(image)
cv2.imwrite("t1_output.jpg", app.draw_on(image, faces))
```

`FaceAnalysis()` defaults to `buffalo_l` and downloads the model package on
first use if needed. Models are stored under `~/.insightface/models/` by
default. `prepare()` uses `ctx_id=0` and Auto detection size, combining
128×128 and 640×640 detection.

### Automatic Provider selection

When no provider is specified, InsightFace selects the first available
provider reported by the installed ONNX Runtime:

```text
CoreMLExecutionProvider → CUDAExecutionProvider → CPUExecutionProvider
```

An accelerated provider uses CPU as its fallback when available. Explicit
`providers=[...]` arguments take precedence. CoreML compilation caches are
reused across runs. See the
[runtime guide](https://github.com/deepinsight/insightface/blob/master/python-package/docs/runtime.md)
for provider overrides, CoreML caching, and telemetry behavior.

## PrivateFrame

PrivateFrame detects and tracks faces in local videos and applies Gaussian
blur or mosaic. Blur all detected faces, blur only people matched to reference
photos, or keep matched people visible. Processing runs locally and preserves
the source video.

```bash
insightface-privateframe process \
  --input /data/video.mp4 --output-dir /data/output
```

This writes `video_privateframe.mp4` and an editable `video_privateframe.json`.
The default **Fast** mode targets **15 analysis FPS**, including in the GUI;
**Normal (30)** provides denser sampling. Analysis FPS controls detection
sampling, not output FPS: every source frame is rendered. Briefly visible
faces can be missed, so review the result before sharing it.

See the [full guide](https://github.com/deepinsight/insightface/blob/master/python-package/insightface/app/privateframe/README.md)
for GUI/Python examples, reference photos, JSON editing, configuration,
automation, and a video demo.

## Evaluation Studio GUI

Install `insightface[gui]`, then launch:

```bash
insightface-gui
```

Evaluation Studio includes PrivateFrame, face comparison and search, People
Library management, album clustering, enterprise evaluation/reporting, and
face swap trials. Workspace data is stored locally and is not uploaded
automatically. See the
[GUI guide](https://github.com/deepinsight/insightface/blob/master/python-package/docs/gui.md)
for model downloads, workflows, settings, and troubleshooting.

## Optional liveness addon

Enable RGB liveness explicitly when constructing `FaceAnalysis`:

```python
import cv2
from insightface.app import FaceAnalysis

app = FaceAnalysis(addons=["liveness"])
app.prepare()
image = cv2.imread("input.jpg")
if image is None:
    raise FileNotFoundError("input.jpg")

for face in app.get(image):
    result = face.liveness
    print(result.status, result.is_live, result.live_score)
```

The addon downloads automatically if missing and is verified before loading.
Its default path is `~/.insightface/addons/liveness.onnx`; enabling it does not
require changing the base model package.

The default `liveness_mode="normal"` keeps detected faces in the results but
skips recognition for faces that fail liveness or have rejected input.
`liveness_mode="observe"` continues recognition regardless of that result.
The default live-score threshold is `0.8`. Omitting `addons=["liveness"]`
disables addon downloading, loading, and inference.

See the [liveness guide](https://github.com/deepinsight/insightface/blob/master/python-package/docs/liveness.md)
for options, result fields, input rejection, offline setup, and error handling.

## Model Zoo

| Workflow | Default model | Alternatives |
|---|---|---|
| `FaceAnalysis()` | `buffalo_l` | Raccoon packages, other supported legacy packs, or your own compatible models |
| PrivateFrame | `raccoon_s` | `raccoon_l` |
| New GUI configurations | `raccoon_s` | Other supported packages for the selected workflow |

Select a package with `FaceAnalysis(name="raccoon_s")`. Model packages live
under `<root>/models/<name>/`; the default root is `~/.insightface`. PrivateFrame
can download its selected Raccoon package on first use. The general GUI model
manager requires an explicit download; existing GUI configurations retain
their saved model selection.

See the [model guide](https://github.com/deepinsight/insightface/blob/master/python-package/docs/model_zoo.md)
for package contents, download links, benchmarks, custom licensed models, and
direct ONNX model calls. Model licenses apply separately from the library's
MIT license.

## Documentation

| Guide | Contents |
|---|---|
| [Runtime and installation](https://github.com/deepinsight/insightface/blob/master/python-package/docs/runtime.md) | Source installs, CUDA, CoreML, provider selection, telemetry |
| [PrivateFrame](https://github.com/deepinsight/insightface/blob/master/python-package/insightface/app/privateframe/README.md) | Video demo, GUI, CLI, Python API, configuration |
| [Liveness](https://github.com/deepinsight/insightface/blob/master/python-package/docs/liveness.md) | Options, results, offline models, input handling |
| [Evaluation Studio](https://github.com/deepinsight/insightface/blob/master/python-package/docs/gui.md) | Desktop workflows and model management |
| [Enterprise evaluation](https://github.com/deepinsight/insightface/blob/master/python-package/docs/commercial_evaluation.md) | Datasets, metrics, and reports |
| [Model Zoo](https://github.com/deepinsight/insightface/blob/master/python-package/docs/model_zoo.md) | Model packages and advanced model usage |
| [Build and packaging](https://github.com/deepinsight/insightface/blob/master/python-package/docs/gui_packaging.md) | Source builds, optional face3d, and distribution |

## Change Log

See the [complete change log](https://github.com/deepinsight/insightface/blob/master/python-package/CHANGELOG.md)
for the September 10, 2026 release notes and earlier versions. Liveness and
PrivateFrame updates are listed separately.
