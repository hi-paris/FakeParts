import argparse
import math
import os
import shutil
import sys
import logging
import torch
from PIL import Image, ImageFilter
import cv2
import decord
import glob
import numpy as np
from torchvision import transforms as T
import torch.nn.functional as F
from einops import rearrange
import traceback
import time
from decord import VideoReader

decord.bridge.set_bridge('torch')
from model.unet_3d_condition_video import UNet3DConditionModel
from diffusers import AutoencoderKL, DDIMScheduler, UNet2DConditionModel, DDPMScheduler, EulerDiscreteScheduler, LMSDiscreteScheduler, EulerAncestralDiscreteScheduler, DPMSolverMultistepScheduler
from diffusers.schedulers.scheduling_pndm import PNDMScheduler
from tqdm.auto import tqdm
from pipelines.stable_diffusion.pipeline_stable_diffusion_video2video_mask_for_video_outpainting import StableDiffusionPipelineVideo2VideoMaskC
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor

import matplotlib.pyplot as plt


class RetryTask(object):

    def __init__(self, retries, raise_if_fail=True):
        self.retries = retries
        self.raise_if_fail = raise_if_fail

    def run(self):
        pass

    def retry_run(self):
        for i in range(self.retries):
            try:
                return self.run()
            except Exception as e:
                traceback.print_exc()
                util_logger.error(
                    "RetryTask run exception: {exc}, try times {time}".format(
                        exc=e, time=i))
                if i == self.retries - 1:
                    if self.raise_if_fail:
                        raise
                    else:
                        return e


def async_thread_tasks(retry_tasks, max_thread_num=16):
    if len(retry_tasks) < 1:
        return []
    if max_thread_num < 1 or max_thread_num > len(retry_tasks):
        max_thread_num = len(retry_tasks)
    with ThreadPoolExecutor(max_thread_num) as thread_pool:
        worker_list = [
            thread_pool.submit(task.retry_run) for task in retry_tasks
        ]
        result = [task.result() for task in worker_list]
        return result


def save_images(test_dir, images, sample_id, stride=1):
    os.makedirs(test_dir, exist_ok=True)
    for i, image in enumerate(images):
        image.save(f"{test_dir}/f{sample_id + i*stride:04d}.jpg")
        print(f"{sample_id + i*stride:04d}.jpg" +"Image sauvegardé")

def count_files_in_directory(directory_path):
    files = [f for f in os.listdir(directory_path) if os.path.isfile(os.path.join(directory_path, f))]
    return len(files)


def preprocess(image, img_transform):
    image = img_transform(image)
    return image


def merge_frame_to_video(frame_dir, fps, output_video_path):
    frame_paths = glob.glob(os.path.join(frame_dir, "*.jpg"))
    frame_paths = sorted(frame_paths)
    fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')

    all_frames = []
    for frame_path in frame_paths:
        frame = cv2.imread(frame_path)
        all_frames.append(frame)

    new_size = (all_frames[0].shape[1], all_frames[0].shape[0])
    new_video = cv2.VideoWriter(output_video_path, fourcc, fps, new_size)
    for high_res_frame in all_frames:
        new_video.write(high_res_frame)
    new_video.release()


def get_max_target_size(cur_shape, target_ratio_list):
    h, w = cur_shape
    cur_ratio = w / h
    max_h, max_w = h, w
    for target_ratio in target_ratio_list:
        target_ratio = float(target_ratio.split(":")[0]) / float(
            target_ratio.split(":")[1])
        if cur_ratio < target_ratio:  # pad w
            max_w = max(max_w, target_ratio * h)
        else:  # pad h
            max_h = max(max_h, w / target_ratio)
    return (max_h, max_w)


def preprocess_frames_speed_square(frames,
                                   target_ratio_list="3:4,1:1,2:3,9:16"):
    res = []

    class preprocess_single_frame_maxsize(RetryTask):

        def __init__(self, frame, index, target_ratio, target_max_size):
            super().__init__(retries=1)
            self.frame = frame
            self.target_ratio = target_ratio
            self.index = index
            self.target_max_size = target_max_size

        def run(self):
            _, h, w = self.frame.shape
            frame_np = self.frame.permute(1, 2, 0).numpy()
            target_height, target_width = self.target_max_size
            max_dim = max(target_height, target_width)
            if max_dim > 256:
                new_size = (256, 256)
            else:
                new_size = (max_dim, max_dim)

            resized_w, resized_h = (int(new_size[0] / max_dim * w),
                                    int(new_size[1] / max_dim * h))

            if resized_h > new_size[1]:
                resized_h = new_size[1]
            if resized_w > new_size[0]:
                resized_w = new_size[0]
            #import pdb;pdb.set_trace()
            frame_np = cv2.resize(frame_np, (resized_w, resized_h))
            h, w, _ = frame_np.shape

            pad_h = (new_size[1] - h) // 2
            pad_w = (new_size[0] - w) // 2
            frame_draw = np.zeros((new_size[1], new_size[0], 3),
                                  dtype=np.uint8)
            frame_draw[pad_h:pad_h + h, pad_w:pad_w + w] = frame_np
            mask = np.ones((new_size[1], new_size[0]), dtype=np.uint8)
            mask[pad_h:pad_h + h, pad_w:pad_w + w] = 0
            frame_inpaint = cv2.inpaint(frame_draw, mask, 5, cv2.INPAINT_TELEA)
            return [self.index, frame_inpaint, mask, frame_draw]

    target_ratio_list = target_ratio_list.split(",")
    _, h, w = frames[0].shape
    target_max_size = get_max_target_size((h, w), target_ratio_list)

    target_ratio = target_max_size[1] / target_max_size[0]
    tasks = []
    for i in range(len(frames)):
        tasks.append(
            preprocess_single_frame_maxsize(frames[i], i, target_ratio,
                                            target_max_size))
    result = async_thread_tasks(tasks, max_thread_num=1)
    result = sorted(result, key=lambda x: x[0])
    frames_inpaint = np.stack([r[1] for r in result], axis=0)
    frames_inpaint_tensor = torch.from_numpy(frames_inpaint).float()
    frames_inpaint_tensor = frames_inpaint_tensor.permute(0, 3, 1,
                                                          2)  # f, c, h, w
    masks = np.stack([r[2] for r in result], axis=0)
    masks_tensor = torch.from_numpy(masks).float().unsqueeze(0)  # [1, n, h, w]
    masked_frames = np.stack([r[3] for r in result], axis=0)
    masked_frames = torch.from_numpy(masked_frames).float()
    masked_frames_tensor = masked_frames.permute(0, 3, 1, 2)  # f, c, h, w

    return frames_inpaint_tensor, masks_tensor, masked_frames_tensor


def merge_frame_to_video_with_ratio(frame_dir, fps, output_video_path,
                                    target_ratios, copy_times=1, fp32 = False):
    if copy_times < 1:
        raise ValueError("copy_times must be >= 1")
    
    frame_paths = glob.glob(os.path.join(frame_dir, "*.jpg"))
    frame_paths = sorted(frame_paths)
    all_frames = [cv2.imread(frame_path) for frame_path in frame_paths]

    if isinstance(target_ratios, str):
        target_ratios = target_ratios.split(',')

    for target_ratio in target_ratios:
        # Convert the string ratio to numerical values
        ratio_parts = target_ratio.split(':')
        target_width_ratio = int(ratio_parts[0])
        target_height_ratio = int(ratio_parts[1])

        # Calculate the target dimensions and crop
        orig_height, orig_width = all_frames[0].shape[:2]
        target_aspect = target_width_ratio / target_height_ratio

        # Determine the necessary crop to achieve desired aspect ratio
        if (orig_width / orig_height) > target_aspect:
            # The width is too wide
            new_width = int(orig_height * target_aspect)
            margin = int((orig_width - new_width) / 2)
            crop_img = (margin, 0, new_width + margin, orig_height)
        else:
            # The height is too tall
            new_height = int(orig_width / target_aspect)
            margin = int((orig_height - new_height) / 2)
            crop_img = (0, margin, orig_width, new_height + margin)

        # Create video writer for each aspect ratio
        fp = 'fp32' if fp32 else 'fp16'
        fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
        video_filename = f"{os.path.splitext(output_video_path)[0]}_{target_ratio.replace(':', '-')}_{fp}.mp4"
        new_video = cv2.VideoWriter(
            video_filename, fourcc, fps,
            (crop_img[2] - crop_img[0], crop_img[3] - crop_img[1]))

        for _ in range(copy_times):
            for high_res_frame in all_frames:
                # Crop the frame according to calculated dimensions
                cropped_frame = high_res_frame[crop_img[1]:crop_img[3],
                                            crop_img[0]:crop_img[2]]
                new_video.write(cropped_frame)

        new_video.release()
        print(f"The video has been saved here :) : {video_filename}!")


def main(args):
    # Print the arguments for verification
    print(args)

    # Set up logging to display useful information
    LOG_FORMAT = "%(asctime)s - %(levelname)s - %(message)s"
    DATE_FORMAT = "%m/%d/%Y %H:%M:%S %p"
    logging.basicConfig(level=logging.INFO,
                        format=LOG_FORMAT,
                        datefmt=DATE_FORMAT)

    # Initialize the execution environment
    device = "cuda"
    weight_dtype = torch.float32 if args.fp32 else torch.float16 # Precision used (float16 for better GPU performance)

    # Read necessary arguments for the pipeline
    input_video_path = args.input_video_path
    gpu_no = args.gpu_no
    pretrained_sd_dir = args.pretrained_sd_dir
    output_dir = args.output_dir
    target_ratio_list = args.target_ratio_list
    video_outpainting_model_dir = args.video_outpainting_model_dir
    seed = args.seed
    copy_times = args.copy_times

     # List to store configured pipelines
    all_pipelines = []

    # Initialize schedulers (PNDM for generation, DDPMScheduler for preprocessing)
    noise_scheduler = PNDMScheduler.from_pretrained(pretrained_sd_dir,
                                                    subfolder="scheduler",
                                                    local_files_only=True)
    scheduler_pre = DDPMScheduler(beta_start=0.00085,
                                  beta_end=0.012,
                                  beta_schedule="scaled_linear",
                                  num_train_timesteps=1000)
    
    # Load components of the pipeline: VAE and UNet3D
    vae = AutoencoderKL.from_pretrained(os.path.join(pretrained_sd_dir, 'vae'),
                                        local_files_only=True,
                                        torch_dtype=weight_dtype)
    print('==== vae ok')
    unet = UNet3DConditionModel.from_pretrained(video_outpainting_model_dir,
                                                local_files_only=True,
                                                torch_dtype=weight_dtype)
    print('==== unet ok')

    # Configure the main pipeline for video outpainting
    pipeline = StableDiffusionPipelineVideo2VideoMaskC(
        vae=vae.to(weight_dtype).to("cuda:%d" % gpu_no).eval(),
        unet=unet.to(weight_dtype).to("cuda:%d" % gpu_no).eval(),
        scheduler=noise_scheduler,
        scheduler_pre=scheduler_pre
        )
    print('==== pipeline ok')
    all_pipelines.append(pipeline)

    # Initial setup for video processing
    num_global_frames = 16  # Number of global frames used for context
    path = input_video_path
    os.makedirs(output_dir, exist_ok=True)

    # Read the video with VideoReader
    vr = VideoReader(open(path, 'rb'))
    fps = vr.get_avg_fps()  # Average framerate
    real_frames_num = len(vr)   # Total number of frames in the video
    total_frames_num = len(vr) * 1
    select_indexes = list(range(0, total_frames_num, 1))    # Indexes of all frames
    print('fps: ', fps, 'total_frames_num: ', total_frames_num)

    # Initialize a list for latents of each frame
    all_latents = [None] * total_frames_num

    # Define the directory to save the generated frames
    frame_dir = os.path.join(output_dir, 'frames')
    filename = os.path.splitext(os.path.basename(input_video_path))[0]
    frame_dir += '_'+ filename
    print("Frame dir is: ", frame_dir)
    if os.path.exists(frame_dir):
        print(f"The folder : {frame_dir} already exist, we delete it")
        shutil.rmtree(frame_dir)
    os.makedirs(frame_dir)
    print(f"The folder :{frame_dir} has been created")

    # Select global frame indexes to provide context
    selected_global_frames_idx = np.linspace(0,
                                             total_frames_num - 1,
                                             num_global_frames,
                                             dtype=int)
    print('global frames:', selected_global_frames_idx)

    print(
        '\'*\' indicates that the frame has been filled, \'|\' indicates it has not been filled.'
    )

    # Define strides based on the total number of frames
    strides = [1] if total_frames_num < 50 else [15, 5, 1]
    
    # Sliding window with different granularities
    for stride in strides:
        stride_indexes = select_indexes[::stride]   # Frames selected for each stride
        
        # Iterate over frames in groups of 16 (local window)
        for start_frame in range(0, (total_frames_num // stride) - 1, 15):

            selected_frame_indexes = stride_indexes[start_frame:start_frame +16]
            duplicate = False
            real_nb_of_frame = len(selected_frame_indexes)

            # Fill with duplicates if the window is incomplete
            if len(selected_frame_indexes) < 16:
                selected_frame_indexes = selected_frame_indexes + [select_indexes[-1]] * (16 - len(selected_frame_indexes))
                assert len(selected_frame_indexes) == 16
                duplicate = True

            print('Frames being filled in:', selected_frame_indexes)

            frames = vr.get_batch(list(map(lambda x: x % real_frames_num, selected_frame_indexes))).permute(0, 3, 1, 2).float()
            global_frames = vr.get_batch(list(map(lambda x: x % real_frames_num,selected_global_frames_idx))).permute(0, 3, 1,2).float()
            img_transform = T.Compose([T.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),])

            # Preprocess global frames
            global_frames = preprocess(global_frames / 255.0,T.Compose([
                    T.Resize((256, 256), antialias=True),
                    T.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
                ]))

            # Preprocess local frames (inpainting and masking)
            frames_inpaint_tensor, mask, masked_frames = preprocess_frames_speed_square(
                frames,
                target_ratio_list=target_ratio_list)  # 3:4,1:1,2:3,9:16

            init_frames = preprocess(
                frames_inpaint_tensor / 255.,
                T.Compose([
                    T.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
                ]))
            init_frames = rearrange(init_frames, "f c h w ->c f h w")

             # Transform masked frames for the model
            masked_frames = preprocess(
                masked_frames / 255.,
                T.Compose([
                    T.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
                ]))
            ori_masked_frames = masked_frames.clone().cpu().numpy()
            masked_frames = rearrange(masked_frames, "f c h w ->c f h w")


            # Move tensors to the GPU
            init_frames = init_frames.to(weight_dtype).to(
                'cuda:{}'.format(gpu_no))
            masked_frames = masked_frames.to(weight_dtype).to(
                'cuda:{}'.format(gpu_no))
            mask = mask.to(weight_dtype).to('cuda:{}'.format(gpu_no))
            global_frames = global_frames.to(weight_dtype).to(
                'cuda:{}'.format(gpu_no))
            generator = torch.Generator(
                device='cuda:{}'.format(gpu_no)).manual_seed(seed)

            # Run outpainting inference
            videos, decoded_latents = all_pipelines[
                0].outpainting_with_random_masked_latent_inference_bidirection(
                    ori_image=init_frames,
                    use_add_noise=True,  # False
                    mask_image=masked_frames,
                    mask=mask[:, :1, :, :],
                    cond=True,
                    strength=1.0,
                    batch_size=1,
                    num_frames=16,
                    height=init_frames.shape[-2],
                    width=init_frames.shape[-1],
                    fps=fps // stride,
                    generator=generator,
                    latents_dtype=weight_dtype,
                    num_inference_steps=50,
                    mask_ratio=6 / 16,
                    copy_raw_images=False,
                    guidance_scale=2,
                    previous_guidance_scale=4.0,
                    cur_step=start_frame // 16,
                    noise_level=100 if stride > 1 else 0,
                    already_outpainted_latents=[
                        all_latents[i_frame]
                        for i_frame in selected_frame_indexes
                    ],
                    copy_already_frame=False,
                    global_frames=global_frames,
                    num_global_frames=num_global_frames
                )  # global_frames_tensor
            
            print("Len de videos : ", len(videos))
            # Merge the results with the original frames
            for k in range(len(videos)):
                images = videos[k].images
                print("Len of images : ", len(images))
                temp_images = []
                for idx in range(real_nb_of_frame):
                    a = (((ori_masked_frames[idx] / 2) + 0.5) *
                        255) * (1 - mask.cpu().numpy())[0, 0]
                    b = np.asarray(images[idx]).transpose(
                        2, 0, 1) * mask.cpu().numpy()[0, 0]
                    temp_images.append(
                        Image.fromarray((a + b).transpose(1, 2,
                                                        0).astype(np.uint8)))
                print("duplicate : ", duplicate)
                print("real_nb_of_frame : ", real_nb_of_frame)
                save_images(frame_dir,
                            images=temp_images,
                            sample_id=selected_frame_indexes[0],
                            stride=stride)

                # Update latents for the generated frames
                for ii in range(len(images)):
                    if all_latents[selected_frame_indexes[ii]] == None:
                        all_latents[selected_frame_indexes[
                            ii]] = decoded_latents[:, :, ii, :, :]
                    else:
                        pass
    
    # Combine frames into the final video
    merge_frame_to_video_with_ratio(
        frame_dir, fps,
        os.path.join(output_dir,
                     '{}.mp4'.format(os.path.basename(path).split(".")[0])),
        target_ratio_list, copy_times, args.fp32)

    #Verify that the number of frame in the directory is equal to the total number of frame
    nb_frame_in_frame_dir = count_files_in_directory(frame_dir)
    if nb_frame_in_frame_dir == total_frames_num :
        print("The video contains exactly all the input frames. Number of frames : ", nb_frame_in_frame_dir)
    elif nb_frame_in_frame_dir > total_frames_num :
        print("There has been an error somewhere, additional frame have added. Number of frames : ", nb_frame_in_frame_dir)
    else :
        print("There has been an error somewhere, frames have deleted. Number of frames : ", nb_frame_in_frame_dir)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Process video with given parameters.")

    # parser.add_argument('--weight_dtype', type = str, default = 'torch.float16')
    parser.add_argument('--input_video_path', type=str, required=True)
    parser.add_argument('--gpu_no', type=int, default=0)
    parser.add_argument('--pretrained_sd_dir', type=str, required=True)
    parser.add_argument('--output_dir', type=str, required=True)
    parser.add_argument('--target_ratio_list', type=str, default='9:16,16:9')
    parser.add_argument('--video_outpainting_model_dir',
                        type=str,
                        required=True)
    parser.add_argument('--copy_times', type=int, default=1, help='Number of loop of the video')
    parser.add_argument('--fp32', action='store_true', default = False, help='Use fp32 (single precision) during inference. Default: fp16 (half precision).')

    parser.add_argument('--seed', type=int, default=6)

    args = parser.parse_args()
    main(args)
