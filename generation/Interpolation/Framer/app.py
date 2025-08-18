import spaces
import datetime
import uuid
from PIL import Image
import numpy as np
import cv2
from scipy.interpolate import interp1d, PchipInterpolator
from packaging import version

import torch
import torchvision
import gradio as gr
# from moviepy.editor import *
from diffusers.utils.import_utils import is_xformers_available
from diffusers.utils import load_image, export_to_video, export_to_gif

import os
import sys
sys.path.insert(0, os.getcwd())
from models_diffusers.controlnet_svd import ControlNetSVDModel
from models_diffusers.unet_spatio_temporal_condition import UNetSpatioTemporalConditionModel
from pipelines.pipeline_stable_video_diffusion_interp_control import StableVideoDiffusionInterpControlPipeline
from gradio_demo.utils_drag import *

import warnings
print("gr file", gr.__file__)

# from huggingface_hub import hf_hub_download, snapshot_download

# os.makedirs("checkpoints", exist_ok=True)

# snapshot_download(
#     "wwen1997/framer_512x320",
#     local_dir="checkpoints/framer_512x320",
#     token=os.environ["TOKEN"],
# )

# snapshot_download(
#     "stabilityai/stable-video-diffusion-img2vid-xt",
#     local_dir="checkpoints/stable-video-diffusion-img2vid-xt",
#     token=os.environ["TOKEN"],
# )


def get_args():
    import argparse
    parser = argparse.ArgumentParser()

    parser.add_argument("--min_guidance_scale", type=float, default=1.0)
    parser.add_argument("--max_guidance_scale", type=float, default=3.0)
    parser.add_argument("--middle_max_guidance", type=int, default=0, choices=[0, 1])
    parser.add_argument("--with_control", type=int, default=1, choices=[0, 1])

    parser.add_argument("--controlnet_cond_scale", type=float, default=1.0)

    parser.add_argument(
        "--dataset",
        type=str,
        default='videoswap',
    )

    parser.add_argument(
        "--model", type=str,
        default="checkpoints/framer_512x320",
        help="Path to model.",
    )

    parser.add_argument("--output_dir", type=str, default="outputs", help="Path to the output video.")

    parser.add_argument("--seed", type=int, default=42, help="random seed.")

    parser.add_argument("--noise_aug", type=float, default=0.02)

    parser.add_argument("--num_frames", type=int, default=14)
    parser.add_argument("--frame_interval", type=int, default=2)

    parser.add_argument("--width", type=int, default=512)
    parser.add_argument("--height", type=int, default=320)

    parser.add_argument(
        "--num_workers",
        type=int,
        default=0,
        help=(
            "Number of subprocesses to use for data loading. 0 means that the data will be loaded in the main process."
        ),
    )

    args = parser.parse_args()

    return args


def interpolate_trajectory(points, n_points):
    x = [point[0] for point in points]
    y = [point[1] for point in points]

    t = np.linspace(0, 1, len(points))

    # fx = interp1d(t, x, kind='cubic')
    # fy = interp1d(t, y, kind='cubic')
    fx = PchipInterpolator(t, x)
    fy = PchipInterpolator(t, y)

    new_t = np.linspace(0, 1, n_points)

    new_x = fx(new_t)
    new_y = fy(new_t)
    new_points = list(zip(new_x, new_y))

    return new_points


def gen_gaussian_heatmap(imgSize=200):
    circle_img = np.zeros((imgSize, imgSize), np.float32)
    circle_mask = cv2.circle(circle_img, (imgSize//2, imgSize//2), imgSize//2, 1, -1)

    isotropicGrayscaleImage = np.zeros((imgSize, imgSize), np.float32)

    for i in range(imgSize):
        for j in range(imgSize):
            isotropicGrayscaleImage[i, j] = 1 / 2 / np.pi / (40 ** 2) * np.exp(
                -1 / 2 * ((i - imgSize / 2) ** 2 / (40 ** 2) + (j - imgSize / 2) ** 2 / (40 ** 2)))

    isotropicGrayscaleImage = isotropicGrayscaleImage * circle_mask
    isotropicGrayscaleImage = (isotropicGrayscaleImage / np.max(isotropicGrayscaleImage)).astype(np.float32)
    isotropicGrayscaleImage = (isotropicGrayscaleImage / np.max(isotropicGrayscaleImage)*255).astype(np.uint8)

    return isotropicGrayscaleImage


def get_vis_image(
        target_size=(512 , 512), points=None,  side=20,
        num_frames=14,
        # original_size=(512 , 512), args="", first_frame=None, is_mask = False, model_id=None,
    ):

    # images = []
    vis_images = []
    heatmap = gen_gaussian_heatmap()

    trajectory_list = []
    radius_list = []
    
    for index, point in enumerate(points):
        trajectories = [[int(i[0]), int(i[1])] for i in point]
        trajectory_list.append(trajectories)

        radius = 20
        radius_list.append(radius)  

    if len(trajectory_list) == 0:
        vis_images = [Image.fromarray(np.zeros(target_size, np.uint8)) for _ in range(num_frames)]
        return vis_images

    for idxx, point in enumerate(trajectory_list[0]):
        new_img = np.zeros(target_size, np.uint8)
        vis_img = new_img.copy()
        # ids_embedding = torch.zeros((target_size[0], target_size[1], 320))
        
        if idxx >= args.num_frames:
            break

        # for cc, (mask, trajectory, radius) in enumerate(zip(mask_list, trajectory_list, radius_list)):
        for cc, (trajectory, radius) in enumerate(zip(trajectory_list, radius_list)):
            
            center_coordinate = trajectory[idxx]
            trajectory_ = trajectory[:idxx]
            side = min(radius, 50)
 
            y1 = max(center_coordinate[1] - side,0)
            y2 = min(center_coordinate[1] + side, target_size[0] - 1)
            x1 = max(center_coordinate[0] - side, 0)
            x2 = min(center_coordinate[0] + side, target_size[1] - 1)
            
            if x2-x1>3 and y2-y1>3:
                need_map = cv2.resize(heatmap, (x2-x1, y2-y1))
                new_img[y1:y2, x1:x2] = need_map.copy()
                
                if cc >= 0:
                    vis_img[y1:y2,x1:x2] = need_map.copy()
                    if len(trajectory_) == 1:
                        vis_img[trajectory_[0][1], trajectory_[0][0]] = 255
                    else:
                        for itt in range(len(trajectory_)-1):
                            cv2.line(vis_img, (trajectory_[itt][0], trajectory_[itt][1]), (trajectory_[itt+1][0], trajectory_[itt+1][1]), (255, 255, 255), 3)

        img = new_img

        # Ensure all images are in RGB format
        if len(img.shape) == 2:  # Grayscale image
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
            vis_img = cv2.cvtColor(vis_img, cv2.COLOR_GRAY2RGB)
        elif len(img.shape) == 3 and img.shape[2] == 3:  # Color image in BGR format
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            vis_img = cv2.cvtColor(vis_img, cv2.COLOR_BGR2RGB)
            
        # Convert the numpy array to a PIL image
        # pil_img = Image.fromarray(img)
        # images.append(pil_img)
        vis_images.append(Image.fromarray(vis_img))

    return vis_images


def frames_to_video(frames_folder, output_video_path, fps=7):
    frame_files = os.listdir(frames_folder)
    # sort the frame files by their names
    frame_files = sorted(frame_files, key=lambda x: int(x.split(".")[0]))

    video = []
    for frame_file in frame_files:
        frame_path = os.path.join(frames_folder, frame_file)
        frame = torchvision.io.read_image(frame_path)
        video.append(frame)

    video = torch.stack(video)
    video = rearrange(video, 'T C H W -> T H W C')
    torchvision.io.write_video(output_video_path, video, fps=fps)


def save_gifs_side_by_side(
    batch_output,
    validation_control_images,
    output_folder,
    target_size=(512 , 512),
    duration=200,
    point_tracks=None,
):
    flattened_batch_output = batch_output
    def create_gif(image_list, gif_path, duration=100):
        pil_images = [validate_and_convert_image(img, target_size=target_size) for img in image_list]
        pil_images = [img for img in pil_images if img is not None]
        if pil_images:
            pil_images[0].save(gif_path, save_all=True, append_images=pil_images[1:], loop=0, duration=duration)

        # also save all the pil_images
        tmp_folder = gif_path.replace(".gif", "")
        print(tmp_folder)
        ensure_dirname(tmp_folder)
        tmp_frame_list = []
        for idx, pil_image in enumerate(pil_images):
            tmp_frame_path = os.path.join(tmp_folder, f"{idx}.png")
            pil_image.save(tmp_frame_path)
            tmp_frame_list.append(tmp_frame_path)
        
        # also save as mp4
        output_video_path = gif_path.replace(".gif", ".mp4")
        frames_to_video(tmp_folder, output_video_path, fps=7)

    # Creating GIFs for each image list
    timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    gif_paths = []

    for idx, image_list in enumerate([validation_control_images, flattened_batch_output]):

        gif_path = os.path.join(output_folder.replace("vis_gif.gif", ""), f"temp_{idx}_{timestamp}.gif")
        create_gif(image_list, gif_path)
        gif_paths.append(gif_path)

        # also save the point_tracks
        assert point_tracks is not None
        point_tracks_path = gif_path.replace(".gif", ".npy")
        np.save(point_tracks_path, point_tracks.cpu().numpy())

    # Function to combine GIFs side by side
    def combine_gifs_side_by_side(gif_paths, output_path):
        print(gif_paths)
        gifs = [Image.open(gif) for gif in gif_paths]

        # Assuming all gifs have the same frame count and duration
        frames = []
        for frame_idx in range(gifs[-1].n_frames):
            combined_frame = None
            for gif in gifs:
                if frame_idx >= gif.n_frames:
                    gif.seek(gif.n_frames - 1)
                else:
                    gif.seek(frame_idx)
                if combined_frame is None:
                    combined_frame = gif.copy()
                else:
                    combined_frame = get_concat_h(combined_frame, gif.copy(), gap=10)
            frames.append(combined_frame)

        if output_path.endswith(".mp4"):
            video = [torchvision.transforms.functional.pil_to_tensor(frame) for frame in frames]
            video = torch.stack(video)
            video = rearrange(video, 'T C H W -> T H W C')
            torchvision.io.write_video(output_path, video, fps=7)
            print(f"Saved video to {output_path}")
        else:
            frames[0].save(output_path, save_all=True, append_images=frames[1:], loop=0, duration=duration)
        
    # Helper function to concatenate images horizontally
    def get_concat_h(im1, im2, gap=10):
        # # img first, heatmap second
        # im1, im2 = im2, im1

        dst = Image.new('RGB', (im1.width + im2.width + gap, max(im1.height, im2.height)), (255, 255, 255))
        dst.paste(im1, (0, 0))
        dst.paste(im2, (im1.width + gap, 0))
        return dst

    # Helper function to concatenate images vertically
    def get_concat_v(im1, im2):
        dst = Image.new('RGB', (max(im1.width, im2.width), im1.height + im2.height))
        dst.paste(im1, (0, 0))
        dst.paste(im2, (0, im1.height))
        return dst

    # Combine the GIFs into a single file
    combined_gif_path = output_folder
    combine_gifs_side_by_side(gif_paths, combined_gif_path)

    combined_gif_path_v = gif_path.replace(".gif", "_v.mp4")
    ensure_dirname(combined_gif_path_v.replace(".mp4", ""))
    combine_gifs_side_by_side(gif_paths, combined_gif_path_v)

    # # Clean up temporary GIFs
    # for gif_path in gif_paths:
    #     os.remove(gif_path)

    return combined_gif_path


# Define functions
def validate_and_convert_image(image, target_size=(512 , 512)):
    if image is None:
        print("Encountered a None image")
        return None

    if isinstance(image, torch.Tensor):
        # Convert PyTorch tensor to PIL Image
        if image.ndim == 3 and image.shape[0] in [1, 3]:  # Check for CxHxW format
            if image.shape[0] == 1:  # Convert single-channel grayscale to RGB
                image = image.repeat(3, 1, 1)
            image = image.mul(255).clamp(0, 255).byte().permute(1, 2, 0).cpu().numpy()
            image = Image.fromarray(image)
        else:
            print(f"Invalid image tensor shape: {image.shape}")
            return None
    elif isinstance(image, Image.Image):
        # Resize PIL Image
        image = image.resize(target_size)
    else:
        print("Image is not a PIL Image or a PyTorch tensor")
        return None
    
    return image


class Drag:

    @spaces.GPU
    def __init__(self, device, args, height, width, model_length, dtype=torch.float16, use_sift=False):
        self.device = device
        self.dtype = dtype

        unet = UNetSpatioTemporalConditionModel.from_pretrained(
            os.path.join(args.model, "unet"),
            torch_dtype=torch.float16,
            low_cpu_mem_usage=True,
            custom_resume=True,
        )
        unet = unet.to(device, dtype)

        controlnet = ControlNetSVDModel.from_pretrained(
            os.path.join(args.model, "controlnet"),
        )
        controlnet = controlnet.to(device, dtype)

        if is_xformers_available():
            import xformers
            xformers_version = version.parse(xformers.__version__)
            unet.enable_xformers_memory_efficient_attention()
            # controlnet.enable_xformers_memory_efficient_attention()
        else:
            raise ValueError(
                "xformers is not available. Make sure it is installed correctly")

        pipe = StableVideoDiffusionInterpControlPipeline.from_pretrained(
            "checkpoints/stable-video-diffusion-img2vid-xt",
            unet=unet,
            controlnet=controlnet,
            low_cpu_mem_usage=False,
            torch_dtype=torch.float16, variant="fp16", local_files_only=True,
        )
        pipe.to(device)

        self.pipeline = pipe
        # self.pipeline.enable_model_cpu_offload()

        self.height = height
        self.width = width
        self.args = args
        self.model_length = model_length
        self.use_sift = use_sift

    @spaces.GPU
    def run(self, first_frame_path, last_frame_path, tracking_points, controlnet_cond_scale, motion_bucket_id):        
        original_width, original_height = 512, 320  # TODO

        # load_image
        image = Image.open(first_frame_path).convert('RGB')
        width, height = image.size
        image = image.resize((self.width, self.height))

        image_end = Image.open(last_frame_path).convert('RGB')
        image_end = image_end.resize((self.width, self.height))

        input_all_points = tracking_points.constructor_args['value']

        sift_track_update = False
        anchor_points_flag = None

        if (len(input_all_points) == 0) and self.use_sift:
            sift_track_update = True
            controlnet_cond_scale = 0.5

            from models_diffusers.sift_match import sift_match
            from models_diffusers.sift_match import interpolate_trajectory as sift_interpolate_trajectory

            output_file_sift = os.path.join(args.output_dir,  "sift.png")

            # (f, topk, 2), f=2 (before interpolation)
            pred_tracks = sift_match(
                image,
                image_end,
                thr=0.5,
                topk=5,
                method="random",
                output_path=output_file_sift,
            )

            if pred_tracks is not None:
                # interpolate the tracks, following draganything gradio demo
                pred_tracks = sift_interpolate_trajectory(pred_tracks, num_frames=self.model_length)

                anchor_points_flag = torch.zeros((self.model_length, pred_tracks.shape[1])).to(pred_tracks.device)
                anchor_points_flag[0] = 1
                anchor_points_flag[-1] = 1

                pred_tracks = pred_tracks.permute(1, 0, 2)  # (num_points, num_frames, 2)

        else:

            resized_all_points = [
                tuple([
                    tuple([int(e1[0] * self.width / original_width), int(e1[1] * self.height / original_height)]) 
                    for e1 in e]) 
                for e in input_all_points
            ]

            # a list of num_tracks tuples, each tuple contains a track with several points, represented as (x, y)
            # in image w & h scale

            for idx, splited_track in enumerate(resized_all_points):
                if len(splited_track) == 0:
                    warnings.warn("running without point trajectory control")
                    continue

                if len(splited_track) == 1: # stationary point
                    displacement_point = tuple([splited_track[0][0] + 1, splited_track[0][1] + 1])
                    splited_track = tuple([splited_track[0], displacement_point])
                # interpolate the track
                splited_track = interpolate_trajectory(splited_track, self.model_length)
                splited_track = splited_track[:self.model_length]
                resized_all_points[idx] = splited_track

            pred_tracks = torch.tensor(resized_all_points)  # (num_points, num_frames, 2)

        vis_images = get_vis_image(
            target_size=(self.args.height, self.args.width),
            points=pred_tracks,
            num_frames=self.model_length,
        )

        if len(pred_tracks.shape) != 3:
            print("pred_tracks.shape", pred_tracks.shape)
            with_control = False
            controlnet_cond_scale = 0.0
        else:
            with_control = True
            pred_tracks = pred_tracks.permute(1, 0, 2).to(self.device, self.dtype)  # (num_frames, num_points, 2)

        point_embedding = None
        video_frames = self.pipeline(
            image,
            image_end,
            # trajectory control
            with_control=with_control,
            point_tracks=pred_tracks,
            point_embedding=point_embedding,
            with_id_feature=False,
            controlnet_cond_scale=controlnet_cond_scale,
            # others
            num_frames=14,
            width=width,
            height=height,
            # decode_chunk_size=8, 
            # generator=generator,
            motion_bucket_id=motion_bucket_id,
            fps=7,
            num_inference_steps=30,
            # track
            sift_track_update=sift_track_update,
            anchor_points_flag=anchor_points_flag,
        ).frames[0]

        vis_images = [cv2.applyColorMap(np.array(img).astype(np.uint8), cv2.COLORMAP_JET) for img in vis_images]
        vis_images = [cv2.cvtColor(np.array(img).astype(np.uint8), cv2.COLOR_BGR2RGB) for img in vis_images]
        vis_images = [Image.fromarray(img) for img in vis_images]
    
        # video_frames = [img for sublist in video_frames for img in sublist]
        val_save_dir = os.path.join(args.output_dir, "vis_gif.gif")
        save_gifs_side_by_side(
            video_frames, 
            vis_images[:self.model_length],
            val_save_dir,
            target_size=(self.width, self.height),
            duration=110,
            point_tracks=pred_tracks,
        )

        return val_save_dir


def reset_states(first_frame_path, last_frame_path, tracking_points):
    first_frame_path = gr.State()
    last_frame_path = gr.State()
    tracking_points = gr.State([])

    return first_frame_path, last_frame_path, tracking_points


def preprocess_image(image):

    image_pil = image2pil(image.name)

    raw_w, raw_h = image_pil.size
    # resize_ratio = max(512 / raw_w, 320 / raw_h)
    # image_pil = image_pil.resize((int(raw_w * resize_ratio), int(raw_h * resize_ratio)), Image.BILINEAR)
    # image_pil = transforms.CenterCrop((320, 512))(image_pil.convert('RGB'))
    image_pil = image_pil.resize((512, 320), Image.BILINEAR)

    first_frame_path = os.path.join(args.output_dir, f"first_frame_{str(uuid.uuid4())[:4]}.png")
    
    image_pil.save(first_frame_path)

    return first_frame_path, first_frame_path, gr.State([])


def preprocess_image_end(image_end):

    image_end_pil = image2pil(image_end.name)

    raw_w, raw_h = image_end_pil.size
    # resize_ratio = max(512 / raw_w, 320 / raw_h)
    # image_end_pil = image_end_pil.resize((int(raw_w * resize_ratio), int(raw_h * resize_ratio)), Image.BILINEAR)
    # image_end_pil = transforms.CenterCrop((320, 512))(image_end_pil.convert('RGB'))
    image_end_pil = image_end_pil.resize((512, 320), Image.BILINEAR)

    last_frame_path = os.path.join(args.output_dir, f"last_frame_{str(uuid.uuid4())[:4]}.png")

    image_end_pil.save(last_frame_path)

    return last_frame_path, last_frame_path, gr.State([])


def add_drag(tracking_points):
    tracking_points.constructor_args['value'].append([])
    return tracking_points


def delete_last_drag(tracking_points, first_frame_path, last_frame_path):
    tracking_points.constructor_args['value'].pop()
    transparent_background = Image.open(first_frame_path).convert('RGBA')
    transparent_background_end = Image.open(last_frame_path).convert('RGBA')
    w, h = transparent_background.size
    transparent_layer = np.zeros((h, w, 4))

    for track in tracking_points.constructor_args['value']:
        if len(track) > 1:
            for i in range(len(track)-1):
                start_point = track[i]
                end_point = track[i+1]
                vx = end_point[0] - start_point[0]
                vy = end_point[1] - start_point[1]
                arrow_length = np.sqrt(vx**2 + vy**2)
                if i == len(track)-2:
                    cv2.arrowedLine(transparent_layer, tuple(start_point), tuple(end_point), (255, 0, 0, 255), 2, tipLength=8 / arrow_length)
                else:
                    cv2.line(transparent_layer, tuple(start_point), tuple(end_point), (255, 0, 0, 255), 2,)
        else:
            cv2.circle(transparent_layer, tuple(track[0]), 5, (255, 0, 0, 255), -1)

    transparent_layer = Image.fromarray(transparent_layer.astype(np.uint8))
    trajectory_map = Image.alpha_composite(transparent_background, transparent_layer)
    trajectory_map_end = Image.alpha_composite(transparent_background_end, transparent_layer)

    return tracking_points, trajectory_map, trajectory_map_end


def delete_last_step(tracking_points, first_frame_path, last_frame_path):
    tracking_points.constructor_args['value'][-1].pop()
    transparent_background = Image.open(first_frame_path).convert('RGBA')
    transparent_background_end = Image.open(last_frame_path).convert('RGBA')
    w, h = transparent_background.size
    transparent_layer = np.zeros((h, w, 4))

    for track in tracking_points.constructor_args['value']:
        if len(track) > 1:
            for i in range(len(track)-1):
                start_point = track[i]
                end_point = track[i+1]
                vx = end_point[0] - start_point[0]
                vy = end_point[1] - start_point[1]
                arrow_length = np.sqrt(vx**2 + vy**2)
                if i == len(track)-2:
                    cv2.arrowedLine(transparent_layer, tuple(start_point), tuple(end_point), (255, 0, 0, 255), 2, tipLength=8 / arrow_length)
                else:
                    cv2.line(transparent_layer, tuple(start_point), tuple(end_point), (255, 0, 0, 255), 2,)
        else:
            cv2.circle(transparent_layer, tuple(track[0]), 5, (255, 0, 0, 255), -1)

    transparent_layer = Image.fromarray(transparent_layer.astype(np.uint8))
    trajectory_map = Image.alpha_composite(transparent_background, transparent_layer)
    trajectory_map_end = Image.alpha_composite(transparent_background_end, transparent_layer)

    return tracking_points, trajectory_map, trajectory_map_end


def add_tracking_points(tracking_points, first_frame_path, last_frame_path, evt: gr.SelectData):  # SelectData is a subclass of EventData
    print(f"You selected {evt.value} at {evt.index} from {evt.target}")
    tracking_points.constructor_args['value'][-1].append(evt.index)

    transparent_background = Image.open(first_frame_path).convert('RGBA')
    transparent_background_end = Image.open(last_frame_path).convert('RGBA')

    w, h = transparent_background.size
    transparent_layer = 0
    for idx, track in enumerate(tracking_points.constructor_args['value']):
        # mask = cv2.imread(
        #     os.path.join(args.output_dir, f"mask_{idx+1}.jpg")
        # )
        mask = np.zeros((320, 512, 3))
        color = color_list[idx+1]
        transparent_layer = mask[:, :, 0].reshape(h, w, 1) * color.reshape(1, 1, -1) + transparent_layer

        if len(track) > 1:
            for i in range(len(track)-1):
                start_point = track[i]
                end_point = track[i+1]
                vx = end_point[0] - start_point[0]
                vy = end_point[1] - start_point[1]
                arrow_length = np.sqrt(vx**2 + vy**2)
                if i == len(track)-2:
                    cv2.arrowedLine(transparent_layer, tuple(start_point), tuple(end_point), (255, 0, 0, 255), 2, tipLength=8 / arrow_length)
                else:
                    cv2.line(transparent_layer, tuple(start_point), tuple(end_point), (255, 0, 0, 255), 2,)
        else:
            cv2.circle(transparent_layer, tuple(track[0]), 5, (255, 0, 0, 255), -1)

    transparent_layer = Image.fromarray(transparent_layer.astype(np.uint8))
    alpha_coef = 0.99
    im2_data = transparent_layer.getdata()
    new_im2_data = [(r, g, b, int(a * alpha_coef)) for r, g, b, a in im2_data]
    transparent_layer.putdata(new_im2_data)

    trajectory_map = Image.alpha_composite(transparent_background, transparent_layer)
    trajectory_map_end = Image.alpha_composite(transparent_background_end, transparent_layer)

    return tracking_points, trajectory_map, trajectory_map_end


if __name__ == "__main__":

    args = get_args()
    ensure_dirname(args.output_dir)
    
    color_list = []
    for i in range(20):
        color = np.concatenate([np.random.random(4)*255], axis=0)
        color_list.append(color)

    with gr.Blocks() as demo:
        gr.Markdown("""<h1 align="center">Framer: Interactive Frame Interpolation</h1><br>""")
    
        gr.Markdown("""Gradio Demo for <a href='https://arxiv.org/abs/2410.18978'><b>Framer: Interactive Frame Interpolation</b></a>.<br>
                    Github Repo can be found at https://github.com/aim-uofa/Framer<br>
                    The template is inspired by DragAnything.""")
    
        gr.Image(label="Framer: Interactive Frame Interpolation", value="assets/demos.gif", height=432, width=768)
    
        gr.Markdown("""## Usage: <br>
                    1. Upload images<br>
                    &ensp;  1.1  Upload the start image via the "Upload Start Image" button.<br>
                    &ensp;  1.2. Upload the end image via the "Upload End Image" button.<br>
                    2. (Optional) Draw some drags.<br>
                    &ensp;  2.1. Click "Add Drag Trajectory" to add the motion trajectory.<br>
                    &ensp;  2.2. You can click several points on either start or end image to forms a path.<br>
                    &ensp;  2.3. Click "Delete last drag" to delete the whole lastest path.<br>
                    &ensp;  2.4. Click "Delete last step" to delete the lastest clicked control point.<br>
                    3. Interpolate the images (according the path) with a click on "Run" button. <br>""")
        
        # device, args, height, width, model_length
        Framer = Drag("cuda", args, 320, 512, 14)
        first_frame_path = gr.State()
        last_frame_path = gr.State()
        tracking_points = gr.State([])
    
        with gr.Row():
            with gr.Column(scale=1):
                image_upload_button = gr.UploadButton(label="Upload Start Image", file_types=["image"])
                image_end_upload_button = gr.UploadButton(label="Upload End Image", file_types=["image"])
                # select_area_button = gr.Button(value="Select Area with SAM")
                add_drag_button = gr.Button(value="Add New Drag Trajectory")
                reset_button = gr.Button(value="Reset")
                run_button = gr.Button(value="Run")
                delete_last_drag_button = gr.Button(value="Delete last drag")
                delete_last_step_button = gr.Button(value="Delete last step")
    
            with gr.Column(scale=7):
                with gr.Row():
                    with gr.Column(scale=6):
                        input_image = gr.Image(
                            label="start frame",
                            interactive=True,
                            height=320,
                            width=512,
                            sources=[],
                        )
    
                    with gr.Column(scale=6):
                        input_image_end = gr.Image(
                            label="end frame",
                            interactive=True,
                            height=320,
                            width=512,
                            sources=[],
                        )
    
        with gr.Row():
            with gr.Column(scale=1):
    
                controlnet_cond_scale = gr.Slider(
                    label='Control Scale', 
                    minimum=0.0, 
                    maximum=10, 
                    step=0.1, 
                    value=1.0,
                )
    
                motion_bucket_id = gr.Slider(
                    label='Motion Bucket', 
                    minimum=1, 
                    maximum=180, 
                    step=1, 
                    value=100,
                )
    
            with gr.Column(scale=5):
                output_video = gr.Image(
                    label="Output Video",
                    height=320,
                    width=1152,
                )
    
    
        with gr.Row():
            gr.Markdown("""
                ## Citation
                ```bibtex
                @article{wang2024framer,
                  title={Framer: Interactive Frame Interpolation},
                  author={Wang, Wen and Wang, Qiuyu and Zheng, Kecheng and Ouyang, Hao and Chen, Zhekai and Gong, Biao and Chen, Hao and Shen, Yujun and Shen, Chunhua},
                  journal={arXiv preprint https://arxiv.org/abs/2410.18978},
                  year={2024}
                }
                ```
                """)
    
        image_upload_button.upload(preprocess_image, image_upload_button, [input_image, first_frame_path, tracking_points])
    
        image_end_upload_button.upload(preprocess_image_end, image_end_upload_button, [input_image_end, last_frame_path, tracking_points])
    
        add_drag_button.click(add_drag, tracking_points, [tracking_points, ])
    
        delete_last_drag_button.click(delete_last_drag, [tracking_points, first_frame_path, last_frame_path], [tracking_points, input_image, input_image_end])
    
        delete_last_step_button.click(delete_last_step, [tracking_points, first_frame_path, last_frame_path], [tracking_points, input_image, input_image_end])
    
        reset_button.click(reset_states, [first_frame_path, last_frame_path, tracking_points], [first_frame_path, last_frame_path, tracking_points])
    
        input_image.select(add_tracking_points, [tracking_points, first_frame_path, last_frame_path], [tracking_points, input_image, input_image_end])
    
        input_image_end.select(add_tracking_points, [tracking_points, first_frame_path, last_frame_path], [tracking_points, input_image, input_image_end])
    
        run_button.click(Framer.run, [first_frame_path, last_frame_path, tracking_points, controlnet_cond_scale, motion_bucket_id], output_video)
    
    demo.launch()
