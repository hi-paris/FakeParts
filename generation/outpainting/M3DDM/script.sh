#!/bin/bash

python src/inference.py --input_video_path "./data/05_default_juggle.mp4" \
--pretrained_sd_dir "./models/SD_model" \
--video_outpainting_model_dir "./models/M3DDM_model" \
--output_dir "./result" \
--target_ratio_list "1:1"

python src/inference.py --input_video_path "./data/bmx-bumps.mp4" \
--pretrained_sd_dir "./models/SD_model" \
--video_outpainting_model_dir "./models/M3DDM_model" \
--output_dir "./result" \
--target_ratio_list "1:1"

python src/inference.py --input_video_path "./data/bmx-trees.mp4" \
--pretrained_sd_dir "./models/SD_model" \
--video_outpainting_model_dir "./models/M3DDM_model" \
--output_dir "./result" \
--target_ratio_list "1:1"

python src/inference.py --input_video_path "./data/breakdance-flare.mp4" \
--pretrained_sd_dir "./models/SD_model" \
--video_outpainting_model_dir "./models/M3DDM_model" \
--output_dir "./result" \
--target_ratio_list "1:1"

python src/inference.py --input_video_path "./data/breakdance-flare.mp4" \
--pretrained_sd_dir "./models/SD_model" \
--video_outpainting_model_dir "./models/M3DDM_model" \
--output_dir "./result" \
--target_ratio_list "1:1" \
--fp32
