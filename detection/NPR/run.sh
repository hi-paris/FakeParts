# Original command for testing predefined datasets

# Test on a custom dataset folder
CUDA_VISIBLE_DEVICES=0 python test.py --model_path NPR_GenImage_sdv4.pth --batch_size 32 --custom_dataset DALLE

CUDA_VISIBLE_DEVICES=0 python test.py --model_path model_epoch_last_3090.pth --batch_size 32 --custom_dataset DALLE
