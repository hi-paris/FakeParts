import torch
import imageio
import os
import argparse
from diffusers.schedulers import EulerAncestralDiscreteScheduler
from transformers import T5EncoderModel, T5Tokenizer
from allegro.pipelines.pipeline_allegro import AllegroPipeline
from allegro.models.vae.vae_allegro import AllegroAutoencoderKL3D
from allegro.models.transformers.transformer_3d_allegro import AllegroTransformer3DModel


def single_inference(args):
    dtype=torch.bfloat16

    # vae have better formance in float32
    vae = AllegroAutoencoderKL3D.from_pretrained(args.vae, torch_dtype=torch.float32).cuda()

    vae.eval()

    text_encoder = T5EncoderModel.from_pretrained(
        args.text_encoder, 
        torch_dtype=dtype
    )
    text_encoder.eval()

    tokenizer = T5Tokenizer.from_pretrained(
        args.tokenizer,
    )

    scheduler = EulerAncestralDiscreteScheduler()

    transformer = AllegroTransformer3DModel.from_pretrained(
        args.dit,
        torch_dtype=dtype
    ).cuda()
    transformer.eval()

    allegro_pipeline = AllegroPipeline(
        vae=vae,
        text_encoder=text_encoder,
        tokenizer=tokenizer,
        scheduler=scheduler,
        transformer=transformer
    ).to("cuda:0")


    positive_prompt = """
(masterpiece), (best quality), (ultra-detailed), (unwatermarked), 
{} 
emotional, harmonious, vignette, 4k epic detailed, shot on kodak, 35mm photo, 
sharp focus, high budget, cinemascope, moody, epic, gorgeous
"""

    negative_prompt = """
nsfw, lowres, bad anatomy, bad hands, text, error, missing fingers, extra digit, fewer digits, cropped, worst quality, 
low quality, normal quality, jpeg artifacts, signature, watermark, username, blurry.
"""

    user_prompt = positive_prompt.format(args.user_prompt.lower().strip())

    if args.enable_cpu_offload:
        allegro_pipeline.enable_sequential_cpu_offload()
        print("cpu offload enabled")
        
    out_video = allegro_pipeline(
        user_prompt, 
        negative_prompt = negative_prompt, 
        num_frames=88,
        height=720,
        width=1280,
        num_inference_steps=args.num_sampling_steps,
        guidance_scale=args.guidance_scale,
        max_sequence_length=512,
        generator = torch.Generator(device="cuda:0").manual_seed(args.seed)
    ).video[0]

    imageio.mimwrite(args.save_path, out_video, fps=15, quality=8)  # highest quality is 10, lowest is 0


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--user_prompt", type=str, default='')
    parser.add_argument("--vae", type=str, default='')
    parser.add_argument("--dit", type=str, default='')
    parser.add_argument("--text_encoder", type=str, default='')
    parser.add_argument("--tokenizer", type=str, default='')
    parser.add_argument("--save_path", type=str, default="./output_videos/test_video.mp4")
    parser.add_argument("--guidance_scale", type=float, default=7.5)
    parser.add_argument("--num_sampling_steps", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--enable_cpu_offload", action='store_true')

    args = parser.parse_args()

    if os.path.dirname(args.save_path) != '' and (not os.path.exists(os.path.dirname(args.save_path))):
        os.makedirs(os.path.dirname(args.save_path))
    
    single_inference(args)
