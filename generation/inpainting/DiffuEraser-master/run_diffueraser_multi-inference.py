import torch
import os 
import time
import argparse
from tqdm import tqdm

from diffueraser.diffueraser import DiffuEraser
from propainter.inference import Propainter, get_device

def davis_folder_style_inference(davis_path, output_dir, quality='480p', save_fps=24):
    """
    Processes all subdirectories in a given directory by calling a script for each subdirectory.
    
    Args:
        davis_path (str): Path to the DAVIS dataset directory.
    
    Returns:
        None
    """
    current_dir = os.path.dirname(os.path.realpath(__file__))
    script_path = os.path.join(current_dir, 'run_diffueraser.py')

    annotations_path = os.path.join(davis_path, 'Annotations/'+ quality)
    jpeg_images_path = os.path.join(davis_path, 'JPEGImages/'+ quality)

    annotations_dir_name = os.listdir(annotations_path)
    
    for video_name in tqdm(annotations_dir_name, desc="Number of videos"):
        masks_dir = os.path.join(annotations_path, video_name)
        frames_dir = os.path.join(jpeg_images_path, video_name)
        
        try:
            command = [
            'python', script_path,  # Assurez-vous d'avoir le bon chemin vers le script run.py
            '--input_video', frames_dir,
            '--input_mask', masks_dir,
            '--max_img_size', '1920',
            '--save_path', output_dir]
            subprocess.run(command)
        
        except Exception as e:
            print(f"Error processing video file {video_name}: {e}")
            continue


############################################################################################################

def get_video_list(input_video_dir, start_rank, end_rank):
    input_video_name = os.listdir(input_video_dir)

    # Filtered video with less than 22 frames (DiffuEraser needs at least 22 frames)
    input_video_name_filtered = []
    for video_name in input_video_name:
        number_frame = len(os.listdir(os.path.join(input_video_dir, video_name)))
        if number_frame >= 22 :
            input_video_name_filtered.append(video_name)

    input_video_name = input_video_name_filtered
    assert(start_rank < len(input_video_name))
    input_video_name.sort()
    if end_rank==-1:
        input_video_name = input_video_name[start_rank:]
    else:
        input_video_name =  input_video_name[start_rank:end_rank]
    input_video_list = []
    for name in input_video_name:
        input_video_list.append(os.path.join(input_video_dir, name))
    return input_video_list

def get_mask_list(input_mask_dir, input_video_list):
    input_mask_list = []
    for input_video in input_video_list:
        input_video_name = os.path.basename(input_video)
        if not(os.path.isdir(input_video)):
            input_video_name=input_video_name[:-4] # get rid of the extension
        input_mask_list.append(os.path.join(input_mask_dir, input_video_name))
    return input_mask_list


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_video_dir', type=str, default="examples/example1/video.mp4", help='Path to the input video')
    parser.add_argument('--input_mask_dir', type=str, default="examples/example1/mask.mp4" , help='Path to the input mask')
    parser.add_argument('--save_path', type=str, default="results" , help='Path to the output')
    parser.add_argument('--save_fps', type=int, default=24, help='fps of the output video')
    parser.add_argument('--start_rank', type=int, default=0, help='Rank of the first video to process. Default: 0.')
    parser.add_argument('--end_rank', type=int, default=100, help='Rank of the last video to process. \'-1\' to select all videos in the folder. Default: 100.')

    parser.add_argument('--video_length', type=int, default=10, help='The maximum length of output video')
    parser.add_argument('--mask_dilation_iter', type=int, default=8, help='Adjust it to change the degree of mask expansion')
    parser.add_argument('--max_img_size', type=int, default=1920, help='The maximum length of output width and height')
    parser.add_argument('--ref_stride', type=int, default=10, help='Propainter params')
    parser.add_argument('--neighbor_length', type=int, default=10, help='Propainter params')
    parser.add_argument('--subvideo_length', type=int, default=50, help='Propainter params')
    parser.add_argument('--base_model_path', type=str, default="weights/stable-diffusion-v1-5" , help='Path to sd1.5 base model')
    parser.add_argument('--vae_path', type=str, default="weights/sd-vae-ft-mse" , help='Path to vae')
    parser.add_argument('--diffueraser_path', type=str, default="weights/diffuEraser" , help='Path to DiffuEraser')
    parser.add_argument('--propainter_model_dir', type=str, default="weights/propainter" , help='Path to priori model')
    args = parser.parse_args()

    # Setup path
    script_dir = os.path.dirname(os.path.realpath(__file__))
    os.chdir(script_dir)

    if not os.path.exists(args.save_path):
        os.makedirs(args.save_path)                        


    start_time = time.perf_counter()
    ## model initialization
    device = get_device()
    # PCM params
    ckpt = "2-Step"
    video_inpainting_sd = DiffuEraser(device, args.base_model_path, args.vae_path, args.diffueraser_path, ckpt=ckpt)
    propainter = Propainter(args.propainter_model_dir, device=device)

    input_video_list = get_video_list(args.input_video_dir, args.start_rank, args.end_rank)
    input_mask_list = get_mask_list(args.input_mask_dir, input_video_list)

    assert(len(input_video_list) == len(input_mask_list))

    for i in tqdm(range(len(input_video_list)), desc="Number of video"):
        input_video = input_video_list[i]
        input_mask = input_mask_list[i]

        base_name = os.path.basename(input_video)
        if os.path.isdir(input_video):
            base_name = base_name+".mp4"
        video_name = base_name
        output_path = os.path.join(args.save_path, video_name)
        priori_path = os.path.join(args.save_path, video_name+"_priori.mp4")

        print("input_video : ", input_video)
        print("input_mask : ",input_mask)
        ## priori
        propainter.forward(input_video, input_mask, priori_path, video_length=args.video_length, 
                            ref_stride=args.ref_stride, neighbor_length=args.neighbor_length, subvideo_length = args.subvideo_length,
                            mask_dilation = args.mask_dilation_iter, save_fps=args.save_fps) 

        ## diffueraser
        guidance_scale = None    # The default value is 0.  
        video_inpainting_sd.forward(input_video, input_mask, priori_path, output_path,
                                    max_img_size = args.max_img_size, video_length=args.video_length, mask_dilation_iter=args.mask_dilation_iter,
                                    guidance_scale=guidance_scale, save_fps=args.save_fps)

    torch.cuda.synchronize() # wait for GPU to finish
    end_time = time.perf_counter()
    inference_time = end_time - start_time  
    print(f"DiffuEraser inference time: {inference_time:.4f} s")

    torch.cuda.empty_cache()

if __name__ == '__main__':
    main()