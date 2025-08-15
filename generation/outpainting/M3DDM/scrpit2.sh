#!/bin/bash

python src/inference.py --input_video_path "./data/dog-gooses.mp4" \
--pretrained_sd_dir "./models/SD_model" \
--video_outpainting_model_dir "./models/M3DDM_model" \
--output_dir "./result" \
--target_ratio_list "1:1"

python src/inference.py --input_video_path "./data/flamingo.mp4" \
--pretrained_sd_dir "./models/SD_model" \
--video_outpainting_model_dir "./models/M3DDM_model" \
--output_dir "./result" \
--target_ratio_list "1:1"

python src/inference.py --input_video_path "./data/flamingo.mp4" \
--pretrained_sd_dir "./models/SD_model" \
--video_outpainting_model_dir "./models/M3DDM_model" \
--output_dir "./result" \
--target_ratio_list "1:1"\
--fp32

python src/inference.py --input_video_path "./data/parkour.mp4" \
--pretrained_sd_dir "./models/SD_model" \
--video_outpainting_model_dir "./models/M3DDM_model" \
--output_dir "./result" \
--target_ratio_list "1:1"

python src/inference.py --input_video_path "./data/tennis.mp4" \
--pretrained_sd_dir "./models/SD_model" \
--video_outpainting_model_dir "./models/M3DDM_model" \
--output_dir "./result" \
--target_ratio_list "1:1"

python src/inference.py --input_video_path "./data/tennis.mp4" \
--pretrained_sd_dir "./models/SD_model" \
--video_outpainting_model_dir "./models/M3DDM_model" \
--output_dir "./result" \
--target_ratio_list "1:1" \
--fp32
