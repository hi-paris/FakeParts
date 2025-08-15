#!/bin/bash

#SBATCH --job-name=gd_sam_paligemma
#SBATCH --output=logs/grounded_sam_2/%j.out
#SBATCH --error=logs/grounded_sam_2/%j.err
#SBATCH --time=24:00:00
#SBATCH --partition=A100
#SBATCH --gpus=1
#SBATCH --cpus-per-task=6

#show nvidia-smi
nvidia-smi

CONDA_ENV="grounded-sam-2"

# Vérification si Conda est installé
if command -v conda &> /dev/null
then
    eval "$(conda shell.bash hook)"
    conda activate ${CONDA_ENV}
    echo "Conda environment activated"
else
    echo "Conda is not installed."
    exit 1
fi

#show conda environment
conda info --envs

# Set cuda version >= 12.0
export PATH=/usr/local/cuda-12.5/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda-12.5/lib64:$LD_LIBRARY_PATH
export CUDA_HOME=/usr/local/cuda-12.5

# Vérification de la version CUDA
if nvcc --version &> /dev/null
then
    echo "CUDA version:"
    nvcc --version
else
    echo "CUDA is not installed or not found."
fi

#setup variables
RUN_SCRIPT="Grounded-SAM-2/automated_video_paligemma2_v1.py"
OUTPUT_DIR="output/propainter"
VIDEO_DIR="/home/ids/saimeur-22/Projects/Gen-Ai/videos/input"

# Echo the command  
set -x

# Run the distributed training script with the necessary parameters
srun python ${RUN_SCRIPT} --video_dir ${VIDEO_DIR} --saving_dir ${OUTPUT_DIR} 