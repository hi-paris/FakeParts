from einops import rearrange
import torch
import imageio
import os
import argparse
from allegro.models.vae.vae_allegro import AllegroAutoencoderKL3D

from decord import VideoReader

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

def vae_inference(args):

    # vae have better performance in float32
    vae = AllegroAutoencoderKL3D.from_pretrained(args.vae, torch_dtype=torch.float32).cuda()

    vae.eval()
    
    vr = VideoReader(args.input_video)

    frames = vr.get_batch(range(len(vr))).asnumpy()
    frames = torch.from_numpy(frames).float() / 255.0
    frames = frames * 2.0 - 1.0
    frames = rearrange(frames, 'f h w c -> 1 c f h w')
    frames = frames[:,:,:88]

    frames = frames.cuda().to(torch.float32)
    with torch.no_grad():
        out_video = vae(frames, encoder_local_batch_size=args.local_batch_size, decoder_local_batch_size=args.local_batch_size).sample
        out_video = ((out_video / 2.0 + 0.5).clamp(0, 1) * 255).to(dtype=torch.uint8).cpu().permute(0, 1, 3, 4, 2).contiguous()

    imageio.mimwrite(f"{args.save_path}/test_vae.mp4", out_video[0], fps=15, quality=8)  # highest quality is 10, lowest is 0



if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--vae", type=str, default='')
    parser.add_argument("--input_video", type=str, default="resources/demo_video.mp4")
    parser.add_argument("--save_path", type=str, default="./output_videos")
    parser.add_argument("--local_batch_size", type=int, default=2)


    args = parser.parse_args()
    if not os.path.exists(args.save_path):
        os.makedirs(args.save_path)
    
    vae_inference(args)
