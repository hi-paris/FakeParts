# Video-Inpainting

## Overview

This repository explores several video inpainting techniques, which involve filling in or "inpainting" specific areas in video frames to conceal objects or regions. The input for these methods includes a video file (e.g., `.mp4`, `.wav`, or a folder of JPEG frames) and an associated mask that identifies the regions to be hidden. To enable inpainting of any object in any video, we use **Grounded-SAM-2** as our mask generator.

We developped a pipeline that take a folder of videos as input and output theses videos with the inpainting of an element of the video.

See this section if your are interessed of reproducing our result and this section to test on your own dataset.

## Contents
- [Grounded SAM 2](#grounded-sam-2)
  - [Grounded SAM 2 Setup](#grounded-sam-2-setup)
  - [Pipeline Overview](#pipeline-overview)
  - [SLURM Inference](#slurm-inference)
  - [Personalized Inference (Paligemma-2)](#personalized-inference-paligemma-2)
- [Propainter](#propainter)
  - [Propainter Setup](#propainter-setup)
  - [Propainter SLURM Script](#propainter-slurm-script)
  - [Propainter Personalized Inference](#propainter-personalized-inference)
- [COCOCO](#cococo)
  - [Setup](#setup)
    - [1. Prerequisites](#1-prerequisites)
    - [2. Package Installation](#2-package-installation)
    - [3. Models' Checkpoints](#3-models-checkpoints)
  - [Example](#example)
- [Citation](#citation) (to complete)



## Grounded-SAM-2

We utilize a modified version of the [Grounded-SAM-2](https://github.com/IDEA-Research/Grounded-SAM-2) package to meet our requirements.

### Grounded-SAM-2 Setup

Setting up the environement and packages.
```bash
conda create -n grounded-sam-2 python=3.10
conda activate grounded-sam-2
```

Installing cuda toolkit 12.1
```bash
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

You have to use CUDA $ \geq $ 12.1. We use cuda 12.5 but the original repo use 12.1 version. 

For example you can set these variables every time you open a new terminal session.
```bash
export PATH=/usr/local/cuda-12.5/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda-12.5/lib64:$LD_LIBRARY_PATH
export CUDA_HOME=/usr/local/cuda-12.5
```

Setting up packages.
```bash
pip install -e .[notebooks,dev]

pip install --no-build-isolation -e grounding_dino

pip install transformers
pip install huggingface-hub==0.25.0 #to avoid [https://github.com/easydiffusion/easydiffusion/issues/1851]
```

Additional packages for Florence-2 (automated prompt generation)
```bash
pip install timm einops
```


### Pipeline Overview

Here's an overview of the pipeline :

The mask generation process follows these steps:

1. **Input**: A directory containing `.mp4` video files.
2. **Frame Selection & Object Identification**:
   - For each video, the first frame is extracted.
   - The extracted frame is passed to **Paligemma2** which identifies the object to be segmented using the following prompt:
     > "What is one interesting object in the image that is neither too small to be noticeable nor so large that it occupies almost the entire frame?"
3. **Object Segmentation**:
   - The identified object is passed as a prompt and segmented across the entire video using **Grounded-SAM-2**.
4. **Output**: The masks are saved in the following directory structure:

```bash
output_folder/ 
├── name_video1/ 
│   └── masks/ 
├── name_video2/ 
│   └── masks/ 
├── ... 
└── name_last_video/
    └── masks/
```

### SLURM Inference

`grounded_sam_2.sh` is the main script to run for automated mask generation.

You need to specify path to your cuda and to your video directory in `grounded_sam_2.sh`.

```
# Set cuda version >= 12.0
export PATH=path/to/cuda/bin:$PATH
export LD_LIBRARY_PATH=path/to/cuda/lib64:$LD_LIBRARY_PATH
export CUDA_HOME=path/to/cuda

#setup variables
VIDEO_DIR="path/to/video/directory"
```

Then, place yourself to the root of the project and run `grounded_sam_2.sh`.

```bash
sbatch ./scripts/propainter_automated/grounded_sam_2_paligemma.sh
```

### Personnalized Inference (Paligemma-2)

To generate masks for all videos in a folder, use `automated_vide_paligemma2_v1.py` with the following command :

```bash
python automated_vide_paligemma2_v1.py \
--video_dir path/to/video/dir
--saving_dir path/to/saving/dir
--start_rank 0 # The rank of the first video, sorted alphabetically. Default: 0 \
--end_rank -1 # The rank of the last video, sorted alphabetically. Use -1 to process all videos. Default: 100
```

To generate a mask for a specific video, use:

```bash
python automated_vide_paligemma2_v1_scripts.py \
--video_path path/to/video/file
--save_tracking_results_dir path/to/saving/result/dir
```

#### Old ReadMe

This package includes two main functions to generate frame-by-frame masks for a specified object:
- `grounded_sam2_tracking_custom_jpeg_video_v2.py`: For inputs as a folder of image frames.
- `grounded_sam2_tracking_custom_video_input_gd1.0_hf_v2.py`: For direct video file inputs.

Both scripts are used in a similar way, with the exception that `grounded_sam2_tracking_custom_jpeg_video_v2.py` requires the directory path of the video frames, while `grounded_sam2_tracking_custom_video_input_gd1.0_hf_v2.py` requires the direct path of the video file.


##### Parameters

Here is a list and brief explanation of the parameters for these functions:

- `-f`, `--frame_dir`: (str) Path to the directory containing video frames (Specific to `grounded_sam2_tracking_custom_jpeg_video_v2.py`).
- `-v`, `--video_path`: (str) Path to the video file (Specific to `grounded_sam2_tracking_custom_video_input_gd1.0_hf_v2.py`).
- `-m`, `--model`: (str) ID of the grounding model. Default is `"IDEA-Research/grounding-dino-tiny"`.
- `-o`, `--output_video_path`: (str) Path to save the output tracking video.
- `-p`, `--prompt_type_for_video`: (str) Type of prompt for the video predictor, choose from `['point', 'box', 'mask']`. Default is `box`.
- `-s`, `--save_tracking_results_dir`: (str) Directory to save tracking results. Default is `"./tracking_results"`.
- `-t`, `--text_prompt`: (str) Text prompt for object grounding.
- `--number`: (int) Maximum number of masks to generate. Set `-1` to include all masks. Default is `1`.
- `--merge`: (bool) Merge masks. Choose `True` or `False`. Default is `True`.

The generated results are saved in the specified tracking results directory (`./tracking_results` by default), containing folders for each mask and other relevant files.

##### Example Command

Here's an example command to generate a mask for a specific object (e.g., a teapot) in a video:

```bash
python grounded_sam2_tracking_custom_video_input_gd1.0_hf_v2.py \
-v ./demo/data/gallery/04_coffee.mp4 \
-t teapot.
```

This command will process the video 04_coffee.mp4 and create a folder result in ./tracking_results, which includes a mask folder for the detected object.


Here's an example command to generate a single mask for sevral specific object (e.g., childrens) in folder frame:

```bash
python grounded_sam2_tracking_custom_jpeg_video_v2.py \
-f ./notebooks/videos/bedroom \
-t children. \
--number -1
```

## Propainter

Currently, we use **ProPainter** for video inpainting, which is the model from [ProPainter: Improving Propagation and Transformer for Video Inpainting](https://shangchenzhou.com/projects/ProPainter/) paper.

The ProPainter paper, published in ICCV 2023, ranks #1 on the YouTube-VOS 2018 benchmark and has gained over 5,000 GitHub stars for its [repository](https://github.com/sczhou/ProPainter).

Refer to Prainter [README](https://github.com/hi-paris/Video-Inpainting/tree/main/Propainter#readme) to understand all the parameters.

### Setup

Setting up the environement and packages.
```bash
conda env create -f environment.yml
conda activate propainter
```

- CUDA >= 9.2


### SLURM

`propainter.sh` is the main script to run for video inpainting.

Ensure your directory structure matches the one described in the [Pipeline Overview](#pipeline-overview) section. 
You also need to specify the path to your video directory and the root directory where the masks from`automated_vide_paligemma2_v1.py` are saved.

```
VIDEO_DIR="path/to/video/directory"
```

Navigate to the root of the project and run `propainter.sh`.

```bash
sbatch ./scripts/propainter_automated/propainter.sh
```

### Personnalized configuration

In the Propainter folder, to generate videos from a directory, use `inference_automated_propainter.py` with the following command :

```bash
python inference_automated_propainter.py \
--directory_path path/to/video/dir
--video_directory path/to/root/of/masks/dir #Cf arborescence of grounded_sam_2.sh
--start_rank 0 # The rank of the first video, sorted alphabetically. Default: 0 \
--end_rank -1 # The rank of the last video, sorted alphabetically. Use -1 to process all videos. Default: 100
```

To generate a mask for a specific video saved in `.mp4` or in `.jpg` frames format, use:

```bash
python inference_propainter.py \
--video path/to/video/file
--mask path/to/mask/direcory
--output path/to/output/folder
```

Test on one video :

```bash
python inference_propainter.py
```

The result will be saved in `Propainter/results/bmx-trees`


## COCOCO

### Setup

#### 1. Prerequisites

-    You have a GPU with at least 24G GPU memory.
-    Your CUDA with nvcc version is greater than 12.0.
-    Your gcc version is greater than 9.4.

#### 2. Package Installation

```bash
conda env create -n cococo python=3.10
conda activate cococo

# Install the CoCoCo dependencies
cd COCOCO-main
pip3 install -r requirements.txt

# Compile the SAM2
pip3 install -e .
```

#### 3. Models's checkpoints

Inside the same folder dedicated to host weights, create two folders : ``cococo_checkpoints`` and ``SD-Inpainting_checkpoints``.
 
In ``cococo_checkpoints`` download [cococo checkpoints](https://mycuhk-my.sharepoint.com/:f:/g/personal/1155203591_link_cuhk_edu_hk/EoXyViqDi8JEgBDCbxsyPY8BCg7YtkOy73SbBY-3WcQ72w?e=cDZuXM)

In ``SD-Inpainting_checkpoints`` download Stable Diffusion Inpainting models. The folder should contains scheduler, tokenizer, text_encoder, vae and unet.

You can use this script to download the Stable Diffusion Inpainting weights.


```bash
from huggingface_hub import snapshot_download

local_dir = "path/to/SD-Inpainting_checkpoints" 
repo_id = "benjamin-paine/stable-diffusion-v1-5-inpainting"

snapshot_download(
    repo_id=repo_id,
    allow_patterns=["scheduler/**", "tokenizer/**", "text_encoder/**", "vae/**", "unet/**"], 
    local_dir=local_dir,
    local_dir_use_symlinks=False
)

print("Download complete")
```

Create a symbolic from ``models_checkpoints`` to folder where there is ``cococo_checkpoints`` and ``SD-Inpainting_checkpoints``

```bash
#Place yourself in COCOCO-main
ln -s path/to/weights/folder/ models_checkpoints 
```

### Example

```bash
python3 valid_code_release.py --config ./configs/code_release.yaml \
--prompt "Man walking along the road. Wind. One mountain in the background. Best quality" \
--negative_prompt "worst quality. bad quality. Trees" \
--guidance_scale 20  \
--video_path ./images/ \
--model_path ./models_checkpoints/cococo_checkpoints \
--pretrain_model_path ./models_checkpoints/SD-Inpainting_checkpoints
```