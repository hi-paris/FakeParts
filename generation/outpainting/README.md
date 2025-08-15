# Video-Outpainting

## M3DDM


### Environment Setup

1. Install PyTorch 2.0.1 with CUDA support via conda:

```sh
conda create -n m3ddm python=3.10
conda activate m3ddm

conda install pytorch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 pytorch-cuda=11.7 -c pytorch -c nvidia
```

Make sure you have Anaconda or Miniconda installed before running this command. This is our testing environment, but it can also run on versions of PyTorch greater than 1.10.0.

2. Install the required dependencies from the `requirements.txt` file in this repository:
```bash
pip install -r requirements.txt
```

### Downloads

Before you can run the project, you need to download the following:

1. **Pre-trained Stable Diffusion Model Weights**:
   
   We used the VAE encoder and decoder inside Stable Diffusion Model. To get the pre-trained stable diffusion v1.5 weights, download them from the following link:
   
   [https://huggingface.co/stable-diffusion-v1-5/stable-diffusion-v1-5](https://huggingface.co/stable-diffusion-v1-5/stable-diffusion-v1-5)



2. **Our Video-Outpainting Model Checkpoints**:
   
   Our network architecture is based on modifications of the Stable Diffusion Model v1.5. To get the pre-trained model weights, download them from the following link:
   
   [https://huggingface.co/alimama-creative/M3DDM-Video-Outpainting](https://huggingface.co/alimama-creative/M3DDM-Video-Outpainting)
   

### Example 

```bash
python src/inference.py --input_video_path "./data/dog-gooses.mp4" \
        --pretrained_sd_dir "./models_checkpoints/sd_checkpoints" \
        --video_outpainting_model_dir "./models_checkpoints/m3ddm_checkpoints" \
        --output_dir "./result" \
        --target_ratio_list "1:1" \
```

### Limitations

**M3DDM** model suffer from blurry result and the output video can't have a dimension higher than 256 (Low quality).

## MOTIA

### Example 

```bash
bash perso_run.sh
``` 

To use your own videos, modify `personalized_exp.yaml` and specify if you want to use `train_outpaint_long.py` (for long videos) or `train_outpaint.py`.

### Limitations

For the moment we hillighted two majors limitations : **The original part is modify** and **the model is time consuming**


| Input Resolution | In. nbr of frames | Output resolution | Out. nbr of frames | gpus | time (min)| Max mem. allocated (GB)
| :---:             | :----:            | :----:            | :----:            | :---:|:----:     |:----:     
| 256 x 256        |  16               | 216 x 512         | 16                 | L40S | 11        | 16.254
| 512 x 512        |  16               | 512  x 1024       | 16                 | L40S | 42        | 36
| 256 x 256        |  128              | 256 x 512         |  64                | L40S | 20        | ...
| 512 x 512        |  128              | 1024 x 512        |  32                | L40S | + 50      | ...


