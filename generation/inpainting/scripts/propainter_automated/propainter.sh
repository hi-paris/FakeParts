#!/bin/bash

#SBATCH --job-name=inpaint-propainter
#SBATCH --output=logs/propainter/%j.out
#SBATCH --error=logs/propainter/%j.err
#SBATCH --time=24:00:00
#SBATCH --partition=A100
#SBATCH --gpus=1
#SBATCH --cpus-per-task=6

#show nvidia-smi
nvidia-smi

CONDA_ENV="propainter"

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

# Set cuda version >= 9.2

# Vérification de la version CUDA
if nvcc --version &> /dev/null
then
    echo "CUDA version:"
    nvcc --version
else
    echo "CUDA is not installed or not found."
fi

#setup variables
RUN_SCRIPT="Propainter/inference_automated_propainter.py"
OUTPUT_DIR="output/propainter"
VIDEO_DIR="/home/ids/saimeur-22/Projects/Gen-Ai/videos/input"

# Echo the command  
set -x

# Run the distributed training script with the necessary parameters
srun python ${RUN_SCRIPT} --directory_path ${OUTPUT_DIR} --video_directory ${VIDEO_DIR}