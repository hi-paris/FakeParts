source ~/.bashrc
conda activate HiFi_Net
CUDA_NUM=1
CUDA_VISIBLE_DEVICES=$CUDA_NUM python HiFi_Net_loc.py 