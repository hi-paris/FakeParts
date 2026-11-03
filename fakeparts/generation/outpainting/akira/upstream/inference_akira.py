import argparse
import json
import os
import torch
import numpy as np
from tqdm import tqdm
from omegaconf import OmegaConf
from PIL import Image
from transformers import CLIPImageProcessor, CLIPVisionModelWithProjection
from diffusers import AutoencoderKLTemporalDecoder, EulerDiscreteScheduler
from diffusers.utils.import_utils import is_xformers_available
from packaging import version as pver

from akira.pipelines.pipeline_animation import StableVideoDiffusionPipelinePoseCond
from akira.models.unet import UNetSpatioTemporalConditionModelPoseCond
from akira.models.pose_adaptor import CameraPoseEncoder
from akira.utils.util import save_videos_grid
from akira.utils.ray_utils import Trajectory
from akira.data.dataset_traj_bokeh_pre_load import RealEstate10KPose

class Args:
    def __init__(self, config):
        for key, value in config.items():
            setattr(self, key, value)

def setup_for_distributed(is_master):
    """
    This function disables printing when not in master process
    """
    import builtins as __builtin__
    builtin_print = __builtin__.print

    def print(*args, **kwargs):
        force = kwargs.pop('force', False)
        if is_master or force:
            builtin_print(*args, **kwargs)

    __builtin__.print = print

def get_pipeline(ori_model_path, unet_subfolder, down_block_types, up_block_types, pose_encoder_kwargs,
                 attention_processor_kwargs, pose_adaptor_ckpt, enable_xformers, device):
    noise_scheduler = EulerDiscreteScheduler.from_pretrained(ori_model_path, subfolder="scheduler")
    feature_extractor = CLIPImageProcessor.from_pretrained(ori_model_path, subfolder="feature_extractor")
    image_encoder = CLIPVisionModelWithProjection.from_pretrained(ori_model_path, subfolder="image_encoder")
    vae = AutoencoderKLTemporalDecoder.from_pretrained(ori_model_path, subfolder="vae")
    unet = UNetSpatioTemporalConditionModelPoseCond.from_pretrained(ori_model_path,
                                                                    subfolder=unet_subfolder,
                                                                    down_block_types=down_block_types,
                                                                    up_block_types=up_block_types)
    pose_encoder = CameraPoseEncoder(**pose_encoder_kwargs)
    print("Setting the attention processors")
    unet.set_pose_cond_attn_processor(enable_xformers=(enable_xformers and is_xformers_available()), **attention_processor_kwargs)
    print(f"Loading weights of camera encoder and attention processor from {pose_adaptor_ckpt}")
    ckpt_dict = torch.load(pose_adaptor_ckpt, map_location=unet.device)
    pose_encoder_state_dict = ckpt_dict['pose_encoder_state_dict']
    pose_encoder_m, pose_encoder_u = pose_encoder.load_state_dict(pose_encoder_state_dict)
    assert len(pose_encoder_m) == 0 and len(pose_encoder_u) == 0
    attention_processor_state_dict = ckpt_dict['attention_processor_state_dict']
    _, attention_processor_u = unet.load_state_dict(attention_processor_state_dict, strict=False)
    assert len(attention_processor_u) == 0
    print("Loading done")
    vae.to(device)
    image_encoder.to(device)
    unet.to(device)
    pipeline = StableVideoDiffusionPipelinePoseCond(
        vae=vae,
        image_encoder=image_encoder,
        unet=unet,
        scheduler=noise_scheduler,
        feature_extractor=feature_extractor,
        pose_encoder=pose_encoder
    )
    pipeline = pipeline.to(device)
    return pipeline


def main(args):
    os.makedirs(os.path.join(args.out_root, 'generated_videos'), exist_ok=True)
    os.makedirs(os.path.join(args.out_root, 'reference_images'), exist_ok=True)
    rank = args.local_rank
    setup_for_distributed(rank == 0)
    gpu_id = rank % torch.cuda.device_count()
    model_configs = OmegaConf.load(args.model_config)
    device = f"cuda:{gpu_id}"
    print(f'Constructing pipeline')
    pipeline = get_pipeline(args.ori_model_path, model_configs['unet_subfolder'], model_configs['down_block_types'],
                            model_configs['up_block_types'], model_configs['pose_encoder_kwargs'],
                            model_configs['attention_processor_kwargs'], args.pose_adaptor_ckpt, args.enable_xformers, device)
    print('Done')

    traj = Trajectory()
    dict_args = {
            "image_width": args.image_width,
            "image_height": args.image_height,
            "video_length": args.num_frames,
            "ori_width": args.original_pose_width,
            "ori_height": args.original_pose_height,
            "gpu_id": 0,
            "is_rectify_intrinsic": True,
            "get_relative_pose": True,
            "plucker_type": args.plucker_type,
        }
    traj_args = Args(dict_args)
    traj.read_from_file(args.trajectory_file, traj_args)
    traj.apply_relative_pose()
    if args.clear_all_pose:
        traj.clear_all_pose()
    
    # traj.rand_intrinsics_distortions(-0.1,-0.1,0.01)
    
    for i, cam in enumerate(traj.cameras):
        cam.oc_info.bokeh_focus_x = args.bokeh_position_x #+ i * 0.05
        cam.oc_info.bokeh_focus_y = args.bokeh_position_y #- i * 0.05
        cam.oc_info.bokeh_K = args.bokeh_K #- i * args.bokeh_K/(args.num_frames-1)
        print(f"bokeh_K: {cam.oc_info.bokeh_K}")
       
        
        cam.oc_info.fx = cam.oc_info.fx * (1.5 - 0.1 * i)
        cam.oc_info.fy = cam.oc_info.fy * (1.5 - 0.1 * i)
        cam.oc_info.fx = cam.oc_info.fx
        cam.oc_info.fy = cam.oc_info.fy
    
    if not args.bokeh:
        print("Computing plucker embedding")
        plucker_embedding = (
            traj.get_plucker_embeddings(device="cpu").squeeze(0).contiguous()
        )
        plucker_embedding = plucker_embedding[None].to(device) 
    else: # bokeh
        print("Computing plucker embedding with **bokeh**")
        plucker_embedding = (
            traj.get_plucker_embeddings_bokehs(device="cpu", bokeh_reprez=args.bokeh_reprez).contiguous()
        )
        list_xy = []
        # visualize the gaussian 
        for i in range(args.num_frames):
            gaussian_manipulate = plucker_embedding[0][i][8] # last emb
            print("min coord: ", np.unravel_index(np.argmin(gaussian_manipulate), gaussian_manipulate.shape)) 
            list_xy.append(np.unravel_index(np.argmin(gaussian_manipulate), gaussian_manipulate.shape))
            # save the gaussian viz in the output
            gaussian_manipulate = gaussian_manipulate.cpu().numpy()
            gaussian_manipulate = gaussian_manipulate / gaussian_manipulate.max()
            import matplotlib.pyplot as plt
            plt.imshow(gaussian_manipulate)
            plt.colorbar()
            folder = f"{args.out_root}/gaussians"
            os.makedirs(folder, exist_ok=True)
            plt.savefig(f"{folder}/gaussian_{i}.png")
            plt.close()
    
    plucker_embedding = plucker_embedding.contiguous().to(device)

    prompt_dict = json.load(open(args.prompt_file, 'r'))
    prompt_images = prompt_dict['image_paths']
    prompt_captions = prompt_dict['captions']
    N = int(len(prompt_images) // args.n_procs)
    remainder = int(len(prompt_images) % args.n_procs)
    prompts_per_gpu = [N + 1 if gpu_id < remainder else N for gpu_id in range(args.n_procs)]
    low_idx = sum(prompts_per_gpu[:gpu_id])
    high_idx = low_idx + prompts_per_gpu[gpu_id]
    prompt_images = prompt_images[low_idx: high_idx]
    prompt_captions = prompt_captions[low_idx: high_idx]
    print(f"rank {rank} / {torch.cuda.device_count()}, number of prompts: {len(prompt_images)}")

    generator = torch.Generator(device=device)
    # generator.manual_seed(42)
    generator.manual_seed(123123)
    for prompt_image, prompt_caption in tqdm(zip(prompt_images, prompt_captions)):
        save_name = "_".join(prompt_caption.split(" "))
        condition_image = Image.open(prompt_image)
        
        # if preprocess image
        if args.preprocess:
            batch_info = {}
            # Convert the PIL image to a NumPy array
            img_np = np.array(condition_image)  # Shape: (H, W, C)
            img_np = img_np[np.newaxis, ...]  # Shape: (1, H, W, C)
            pixel_values = torch.from_numpy(img_np).permute(0, 3, 1, 2).contiguous()  # Shape: (1, C, H, W)
            pixel_values = pixel_values.float() / 255.0
            
            batch_info['pixel_values'] = [pixel_values]
            batch_info['trajs'] = [traj]
            batch_info['depth_imgs'] = [None]
            batch_pixel_values = RealEstate10KPose.get_pre_process_and_augmentation_condition_image(batch_info, True, aug_idx=[0])
            
            # change it back to PIL
            pixel_values = batch_pixel_values[0]
            img_tensor = pixel_values[0]  # Shape: (C, H, W)
            img_tensor = img_tensor.permute(1, 2, 0)  # Shape: (H, W, C)
            img_np = (img_tensor.cpu().numpy() * 255).astype(np.uint8)
            condition_image = Image.fromarray(img_np)
        
        with torch.no_grad():
            sample = pipeline(
                image=condition_image,
                pose_embedding=plucker_embedding,
                height=args.image_height,
                width=args.image_width,
                num_frames=args.num_frames,
                num_inference_steps=args.num_inference_steps,
                min_guidance_scale=args.min_guidance_scale,
                max_guidance_scale=args.max_guidance_scale,
                do_image_process=True,
                generator=generator,
                output_type='pt'
            ).frames[0].transpose(0, 1).cpu()      # [3, f, h, w] 0-1
        resized_condition_image = condition_image.resize((args.image_width, args.image_height))
        
        if args.bokeh:
            pass
            # viz of the bokeh area
            # pix_width = 5
            # for idx, xy in enumerate(list_xy):
                # sample[2, idx, xy[0]-pix_width:xy[0]+pix_width, xy[1]-pix_width:xy[1]+pix_width] = 1.0
        
        save_videos_grid(sample[None], f"{os.path.join(args.out_root, 'generated_videos')}/{save_name}.mp4", rescale=False)
        resized_condition_image.save(os.path.join(args.out_root, 'reference_images', f'{save_name}.png'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--out_root", type=str)
    parser.add_argument("--image_height", type=int, default=320)
    parser.add_argument("--image_width", type=int, default=576)
    parser.add_argument("--num_frames", type=int, default=14, help="14 for svd and 25 for svd-xt", choices=[14, 25])
    parser.add_argument("--ori_model_path", type=str)
    parser.add_argument("--unet_subfolder", type=str, default='unet')
    parser.add_argument("--enable_xformers", action='store_true')
    parser.add_argument("--pose_adaptor_ckpt", default=None)
    parser.add_argument("--num_inference_steps", type=int, default=25)
    parser.add_argument("--min_guidance_scale", type=float, default=1.0)
    parser.add_argument("--max_guidance_scale", type=float, default=3.0)
    parser.add_argument("--prompt_file", required=True, help='prompts path, json or txt')
    parser.add_argument("--trajectory_file", required=True)
    parser.add_argument("--original_pose_width", type=int, default=1280)
    parser.add_argument("--original_pose_height", type=int, default=720)
    parser.add_argument("--model_config", required=True)
    parser.add_argument("--n_procs", type=int, default=8)
    parser.add_argument('--bokeh', action='store_true', help='use bokeh')
    parser.add_argument('--bokeh_K', type=float, default=0.0)
    parser.add_argument('--bokeh_reprez', type=str, default='sigmoid', help='sigmoid, power, inv_power')
    parser.add_argument('--bokeh_position_x', type=float, default=0.0)
    parser.add_argument('--bokeh_position_y', type=float, default=0.0)

    # DDP args
    parser.add_argument("--world_size", default=1, type=int,
                        help="number of the distributed processes.")
    parser.add_argument('--local_rank', type=int, default=-1,
                        help='Replica rank on the current node. This field is required '
                             'by `torch.distributed.launch`.')
    parser.add_argument("--plucker_type", type=str, default="cctrl")
    parser.add_argument("--preprocess", action='store_true')
    parser.add_argument("--clear_all_pose", action='store_true')
    args = parser.parse_args()
    main(args)
