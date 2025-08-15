import os
import math
import wandb
import random
import logging
import inspect
import argparse
import datetime
import subprocess

import cv2
from pathlib import Path
import numpy as np
from tqdm.auto import tqdm
from einops import rearrange
from omegaconf import OmegaConf
from safetensors import safe_open
from typing import Dict, Optional, Tuple

import torch
import torchvision
import torch.nn.functional as F
import torch.distributed as dist
from torch.optim.swa_utils import AveragedModel
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP

import diffusers
from diffusers import AutoencoderKL, DDIMScheduler
from diffusers.models import UNet2DConditionModel
from diffusers.pipelines import StableDiffusionInpaintPipeline
from diffusers.optimization import get_scheduler
from diffusers.utils import check_min_version
from diffusers.utils.import_utils import is_xformers_available

import transformers
from transformers import CLIPTextModel, CLIPProcessor, CLIPVisionModel, CLIPTokenizer

from cococo.models.unet import UNet3DConditionModel
from cococo.pipelines.pipeline_animation_inpainting_cross_attention_vae import AnimationInpaintPipeline
from cococo.utils.util import save_videos_grid, zero_rank_print

# This function is an inference pipeline for a video inpainting model that uses guided diffusion.
# The goal is to take a video with masked regions and a prompt describing the desired content,
# and generate a new video where the masked areas are filled based on the prompt.

def main(
    name: str,                          # Experiment or run name for organizing outputs.
    use_wandb: bool,                    # Whether to use Weights & Biases for logging.
    launcher: str,                      # Launcher type (e.g., for distributed execution).
    model_path: str,                    # Path to the trained model weights.
    prompt: str,                        # Text prompt for guiding video generation.
    negative_prompt: str,               # Negative prompt to avoid certain attributes.
    guidance_scale: float,              # Controls adherence to the prompt; higher = more guidance.
    output_dir: str,                    # Directory for saving outputs.
    pretrained_model_path: str,         # Path to pre-trained components (VAE, tokenizer, etc.).
    sub_folder: str = "unet",           # Subfolder for loading UNet-specific configurations.
    unet_checkpoint_path: str = "",     # Path to UNet checkpoint if different from defaults.
    unet_additional_kwargs: Dict = {},  # Additional arguments for UNet initialization.
    noise_scheduler_kwargs=None,        # Configurations for the noise scheduler.
    num_workers: int = 32,              # Number of workers for data loading.
    enable_xformers_memory_efficient_attention: bool = True,  # Enable memory-efficient attention for faster inference.
    image_path: str = '',               # Path to the input video file (in `.npy` format).
    mask_path: str = '',                # Path to the corresponding mask file (in `.npy` format).
    global_seed: int = 42,              # Seed for reproducibility.
    is_debug: bool = False,             # Debug flag for additional logging.
):
    # Set the random seed for PyTorch to ensure reproducibility.
    seed = global_seed
    torch.manual_seed(seed)

    # Create an output folder named based on the experiment name and timestamp.
    folder_name = name + datetime.datetime.now().strftime("-%Y-%m-%dT%H-%M-%S")
    output_dir = os.path.join(output_dir, folder_name)

    # Save the configuration of this run as a YAML file for reproducibility and logging.
    *_, config = inspect.getargvalues(inspect.currentframe())
    os.makedirs(output_dir, exist_ok=True)
    OmegaConf.save(config, os.path.join(output_dir, 'config.yaml'))
    print("Creation of specific output folder!")
    print("\n   ")

    # Initialize the noise scheduler for the diffusion process based on provided configurations.
    noise_scheduler = DDIMScheduler(**OmegaConf.to_container(noise_scheduler_kwargs))

    # Load pretrained components: VAE, tokenizer, and text encoder.
    print("pretrained_model_path : ", os.path.join(pretrained_model_path, 'vae'))  
    vae = AutoencoderKL.from_pretrained(os.path.join(pretrained_model_path, 'vae'))
    print(" ==== vae ok")
    tokenizer = CLIPTokenizer.from_pretrained(pretrained_model_path, subfolder="tokenizer")
    print(" ==== tokenizer ok")
    text_encoder = CLIPTextModel.from_pretrained(pretrained_model_path, subfolder="text_encoder")
    print(" ====  text_encoder ok")

    # Load the UNet model for conditional generation with additional configurations.
    unet = UNet3DConditionModel.from_pretrained_2d(
        pretrained_model_path, subfolder=sub_folder,
        unet_additional_kwargs=OmegaConf.to_container(unet_additional_kwargs)
    )
    print(" === unet ok")

    # Load the state dictionary of the model weights from multiple checkpoints and merge them.
    state_dict = {}
    for i in range(4):  # Combine weights from 4 separate checkpoints.
        state_dict2 = torch.load(f'{model_path}/model_{i}.pth', map_location='cpu')
        state_dict = {**state_dict, **state_dict2}

    # Filter and adjust keys in the state dictionary for compatibility with the UNet model.
    state_dict2 = {}
    for key in state_dict:
        if 'pe' in key:
            continue
        state_dict2[key.split('module.')[1]] = state_dict[key]

    # Load the state dictionary into the UNet model.
    m, u = unet.load_state_dict(state_dict2, strict=False)

    # Move models to GPU, switch to half-precision for faster inference, and set to evaluation mode.
    vae = vae.cuda().half().eval()
    text_encoder = text_encoder.cuda().half().eval()
    unet = unet.cuda().half().eval()

    # Initialize the inference pipeline using the loaded components.
    validation_pipeline = AnimationInpaintPipeline(
        unet=unet, vae=vae, tokenizer=tokenizer, text_encoder=text_encoder, scheduler=noise_scheduler,
    )
    validation_pipeline.enable_vae_slicing()  # Enable efficient memory slicing for the VAE.

    # Load video frames and masks from the input paths.
    images = 2 * (np.load(image_path) / 255.0 - 0.5)  # Normalize images to [-1, 1].
    masks = np.load(mask_path) / 255.0  # Normalize masks to [0, 1].
    pixel_values = torch.tensor(images).to(device=vae.device, dtype=torch.float16)  # Convert to tensors.
    test_masks = torch.tensor(masks).to(device=vae.device, dtype=torch.float16)

    # Extract video dimensions.
    height, width = images.shape[-3:-1]

    # Define a prefix for saving generated images in the output directory.
    prefix = 'test_release_' + str(guidance_scale) + '_guidance_scale'

    # Convert images and masks to latent space using the VAE.
    latents, masks = [], []
    with torch.no_grad():    # No gradient computation for inference.
        for i in range(len(pixel_values)):
            pixel_value = rearrange(pixel_values[i:i+1], "f h w c -> f c h w")
            test_mask = rearrange(test_masks[i:i+1], "f h w c -> f c h w")

            # Encode images into latents and interpolate masks to latent resolution.
            masked_image = (1 - test_mask) * pixel_value    # Apply mask to image.
            latent = vae.encode(masked_image).latent_dist.sample()  # Encode image into latent space.
            test_mask = torch.nn.functional.interpolate(test_mask, size=latent.shape[-2:]).cuda()

            # Rearrange dimensions for compatibility with the diffusion pipeline.
            latent = rearrange(latent, "f c h w -> c f h w")
            test_mask = rearrange(test_mask, "f c h w -> c f h w")

            latent = latent * 0.18215  # Scale latent values.
            latents.append(latent)
            masks.append(test_mask)

    # Combine latents and masks across all frames.
    latents = torch.cat(latents, dim=1)  # Concatenate all latents along the frame dimension.
    test_masks = torch.cat(masks, dim=1)     # Concatenate all masks similarly.
    latents = latents[None, ...]    # Add a batch dimension to latents.
    masks = test_masks[None, ...]   # Add a batch dimension to masks.

    # Set up a random generator for reproducibility.
    generator = torch.Generator(device=latents.device)
    generator.manual_seed(0)

    # Loop through inference steps to generate video frames.
    for step in range(5):  # Generate 10 steps of video frames.
        with torch.no_grad():  # Disable gradients during inference.
            # Run the pipeline for video generation.
            videos, masked_videos, recon_videos = validation_pipeline(
                prompt,
                image=latents,
                masked_image=latents,
                masked_latents=None,
                masks=masks,
                generator=generator,
                video_length=len(images),
                negative_prompt=negative_prompt,
                height=height,
                width=width,
                num_inference_steps=50,  # Number of diffusion steps.
                guidance_scale=guidance_scale,  # Prompt guidance strength.
            )
        
        # Post-process the generated videos for decoding.
        videos = videos.permute(0, 2, 1, 3, 4).contiguous() / 0.18215  # Rearrange and scale latents.
        
        with torch.no_grad():
            images = []
            for i in range(len(videos[0])):  # Decode each frame.
                image = vae.decode(videos[0][i:i+1].half()).sample  # Decode from VAE latent.
                images.append(image)  # Append decoded frame to list.
            video = torch.cat(images, dim=0)  # Combine frames into a single tensor.
            video = video / 2 + 0.5  # Normalize back to [0, 1].
            video = torch.clamp(video, 0, 1)  # Ensure pixel values are within valid range.
            video = video.permute(0, 2, 3, 1)  # Rearrange dimensions for saving.
        
        # Convert video frames to uint8 and save as individual images.
        video = 255.0 * video.cpu().detach().numpy()  # Scale to [0, 255].
        for i in range(len(video)):
            image = video[i]  # Extract single frame.
            image = np.uint8(image)  # Convert to uint8 format.
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert to RGB.
            cv2.imwrite(output_dir + '/' + prefix + '_' + str(step) + '_image_' + str(i) + '.png', image)  # Save image.

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config",   type=str, required=True)
    parser.add_argument("--prompt", type=str, default="")
    parser.add_argument("--negative_prompt", type=str, default="")
    parser.add_argument("--model_path", type=str, default="../")
    parser.add_argument("--pretrain_model_path", type=str, default="../")
    parser.add_argument("--sub_folder", type=str, default="unet")
    parser.add_argument("--guidance_scale", type=float, default=20)
    parser.add_argument("--video_path", type=str, default="")
    args = parser.parse_args()

    name   = Path(args.config).stem
    config = OmegaConf.load(args.config)

    

    main(name=name, \
        launcher=None, \
        use_wandb=False, \
        prompt=args.prompt, \
        model_path=args.model_path, \
        sub_folder=args.sub_folder, \
        pretrained_model_path=args.pretrain_model_path, \
        negative_prompt=args.negative_prompt, \
        guidance_scale=args.guidance_scale, \
        image_path=args.video_path+'/images.npy', \
        mask_path=args.video_path+'/masks.npy', \
        output_dir='output', \
        **config
        )
