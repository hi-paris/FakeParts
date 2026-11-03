import inspect
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Union

import torch
import numpy as np
import PIL.Image
import torch

from diffusers.image_processor import VaeImageProcessor
from diffusers.utils import BaseOutput, logging
from diffusers.utils.torch_utils import randn_tensor, is_compiled_module
from diffusers.pipelines.pipeline_utils import DiffusionPipeline

logger = logging.get_logger(__name__)

def encode_image_with_id_features(img, id_features, model, dino_layers = 24):
    '''
        img: B, 3, 224, 224
        id_features: B, 1, 512    
    '''
    W, H = img.shape[-2:]
    dino_f = model.patch_embed(img)
    dino_f = torch.cat((model.cls_token.expand(dino_f.shape[0], -1, -1), dino_f), dim=1)
    dino_f = dino_f + model.interpolate_pos_encoding(dino_f, W, H)
    dino_f = torch.cat((
        dino_f[:, :1],
        model.register_tokens.expand(dino_f.shape[0], -1, -1),
        dino_f[:, 1:],
    ), dim=1)
    for i in range(dino_layers):
        dino_f = model.blocks[i](dino_f, id_features)
    dino_f = model.norm(dino_f)
    return dino_f

def tensor2vid(video: torch.Tensor, processor: VaeImageProcessor, output_type: str = "np"):
    batch_size = video.shape[0]
    outputs = []
    for batch_idx in range(batch_size):
        batch_vid = video[batch_idx].permute(1, 0, 2, 3)
        batch_output = processor.postprocess(batch_vid, output_type)

        outputs.append(batch_output)

    if output_type == "np":
        outputs = np.stack(outputs)

    elif output_type == "pt":
        outputs = torch.stack(outputs)

    elif not output_type == "pil":
        raise ValueError(f"{output_type} does not exist. Please choose one of ['np', 'pt', 'pil']")

    return outputs


@dataclass
class HunyuanSVDPipelineOutput(BaseOutput):
    r"""
    Output class for zero-shot text-to-video pipeline.

    Args:
        frames (`[List[PIL.Image.Image]`, `np.ndarray`]):
            List of denoised PIL images of length `batch_size` or NumPy array of shape `(batch_size, height, width,
            num_channels)`.
    """

    frames: Union[List[PIL.Image.Image], np.ndarray]


class HunyuanLongSVDPipeline(DiffusionPipeline):
    r"""
    Pipeline to generate video from an input image using Stable Video Diffusion.

    This model inherits from [`DiffusionPipeline`]. Check the superclass documentation for the generic methods
    implemented for all pipelines (downloading, saving, running on a particular device, etc.).

    """

    model_cpu_offload_seq = "image_encoder->unet->vae"
    _callback_tensor_inputs = ["latents"]

    def __init__(
        self,
        vae,
        image_encoder,
        image_proj,
        unet,
        pose_guider,
        scheduler,
    ):
        super().__init__()
        self.register_modules(
            vae=vae,
            image_encoder=image_encoder,
            image_proj=image_proj,
            unet=unet,
            pose_guider=pose_guider,
            scheduler=scheduler,
        )
        
        self.vae_scale_factor = 2 ** (len(self.vae.config.block_out_channels) - 1)

        self.image_processor = VaeImageProcessor(
            vae_scale_factor=self.vae_scale_factor,
            do_convert_rgb=True)
        
        self.pose_image_processor = VaeImageProcessor(
            vae_scale_factor=self.vae_scale_factor,
            do_convert_rgb=True,
            do_normalize=False,
        )


    def _encode_image(self, image, arcface_embeddings, prompts, uncond_prompts, num_frames, device, num_videos_per_prompt, do_classifier_free_guidance):
        dtype = next(self.image_encoder.parameters()).dtype

        image = image.to(device=device, dtype=dtype)
        if arcface_embeddings is not None:
            arcface_embeddings = torch.from_numpy(arcface_embeddings).to(image)
        
        image_embeddings = encode_image_with_id_features(image, arcface_embeddings.unsqueeze(0), self.image_encoder)
        image_embeddings = self.image_proj(image_embeddings.permute(0, 2, 1)).permute(0, 2, 1)

        # duplicate image embeddings for each generation per prompt, using mps friendly method
        bs_embed, seq_len, _ = image_embeddings.shape
        image_embeddings = image_embeddings.repeat(1, num_videos_per_prompt, 1)
        image_embeddings = image_embeddings.view(bs_embed * num_videos_per_prompt, seq_len, -1)
        
        image_embeddings = image_embeddings.unsqueeze(1).repeat((1, num_frames, 1, 1))
        
        if do_classifier_free_guidance:
            negative_image_embeddings = torch.zeros_like(image_embeddings)
            image_embeddings = torch.cat([negative_image_embeddings, image_embeddings, image_embeddings]).to(device=device, dtype=dtype)
            prompts = torch.cat([uncond_prompts, uncond_prompts, prompts]).to(device=device, dtype=dtype)

        return image_embeddings, prompts

    def _encode_vae_image(
        self,
        image: torch.Tensor,
        device,
        num_videos_per_prompt,
        do_classifier_free_guidance,
    ):
        image = image.to(device=device)
        image_latents = self.vae.encode(image).latent_dist.mode()
        
        if do_classifier_free_guidance:
            negative_image_latents = torch.zeros_like(image_latents)
            image_latents = torch.cat([negative_image_latents, image_latents, image_latents])

        image_latents = image_latents.repeat(num_videos_per_prompt, 1, 1, 1)

        return image_latents
    
    def _get_add_time_ids(
        self,
        fps,
        motion_bucket_id,
        noise_aug_strength,
        dtype,
        batch_size,
        num_videos_per_prompt,
        do_classifier_free_guidance,
    ):
        add_time_ids = [fps, motion_bucket_id, noise_aug_strength]

        passed_add_embed_dim = self.unet.config.addition_time_embed_dim * len(add_time_ids)
        expected_add_embed_dim = self.unet.add_embedding.linear_1.in_features

        if expected_add_embed_dim != passed_add_embed_dim:
            raise ValueError(
                f"Model expects an added time embedding vector of length {expected_add_embed_dim}, but a vector of {passed_add_embed_dim} was created. The model has an incorrect config. Please check `unet.config.time_embedding_type` and `text_encoder_2.config.projection_dim`."
            )

        add_time_ids = torch.tensor([add_time_ids], dtype=dtype)
        add_time_ids = add_time_ids.repeat(batch_size * num_videos_per_prompt, 1)

        if do_classifier_free_guidance:
            add_time_ids = torch.cat([add_time_ids, add_time_ids, add_time_ids])

        return add_time_ids
    
    def decode_latents(self, latents, num_frames, decode_chunk_size=14):
        # [batch, frames, channels, height, width] -> [batch*frames, channels, height, width]
        latents = latents.flatten(0, 1)
        latents = 1 / self.vae.config.scaling_factor * latents

        forward_vae_fn = self.vae._orig_mod.forward if is_compiled_module(self.vae) else self.vae.forward
        accepts_num_frames = "num_frames" in set(inspect.signature(forward_vae_fn).parameters.keys())

        # decode decode_chunk_size frames at a time to avoid OOM
        frames = []
        for i in range(0, latents.shape[0], decode_chunk_size):
            num_frames_in = latents[i : i + decode_chunk_size].shape[0]
            decode_kwargs = {}
            if accepts_num_frames:
                decode_kwargs["num_frames"] = num_frames_in

            frame = self.vae.decode(latents[i : i + decode_chunk_size], **decode_kwargs).sample
            frames.append(frame)
        frames = torch.cat(frames, dim=0)

        # [batch*frames, channels, height, width] -> [batch, channels, frames, height, width]
        frames = frames.reshape(-1, num_frames, *frames.shape[1:]).permute(0, 2, 1, 3, 4)

        # we always cast to float32 as this does not cause significant overhead and is compatible with bfloat16
        frames = frames.float()
        return frames

    def check_inputs(self, image, height, width):
        if (
            not isinstance(image, torch.Tensor)
            and not isinstance(image, PIL.Image.Image)
            and not isinstance(image, list)
        ):
            raise ValueError(
                "`image` has to be of type `torch.FloatTensor` or `PIL.Image.Image` or `List[PIL.Image.Image]` but is"
                f" {type(image)}"
            )

        if height % 8 != 0 or width % 8 != 0:
            raise ValueError(f"`height` and `width` have to be divisible by 8 but are {height} and {width}.")

    def prepare_latents(
        self,
        batch_size,
        num_frames,
        num_channels_latents,
        height,
        width,
        dtype,
        device,
        generator,
        latents=None,
        ref_image_latents=None,
        timestep=None
    ):
        shape = (
            batch_size,
            num_frames,
            num_channels_latents // 2,
            height // self.vae_scale_factor,
            width // self.vae_scale_factor,
        )
        if isinstance(generator, list) and len(generator) != batch_size:
            raise ValueError(
                f"You have passed a list of generators of length {len(generator)}, but requested an effective batch"
                f" size of {batch_size}. Make sure the batch size matches the length of the generators."
            )

        if latents is None:
            noise = randn_tensor(shape, generator=generator, device=device, dtype=dtype)
        else:
            noise = latents.to(device)

        # scale the initial noise by the standard deviation required by the scheduler
        
        if timestep is not None:
            init_latents = ref_image_latents.unsqueeze(1)
            latents = self.scheduler.add_noise(init_latents, noise, timestep)
        else:
            latents = noise * self.scheduler.init_noise_sigma
        return latents
    
    def get_timesteps(self, num_inference_steps, strength, device):
        # get the original timestep using init_timestep
        init_timestep = min(int(num_inference_steps * strength), num_inference_steps)

        t_start = max(num_inference_steps - init_timestep, 0)
        timesteps = self.scheduler.timesteps[t_start * self.scheduler.order :]

        return timesteps, num_inference_steps - t_start

    @property
    def guidance_scale1(self):
        return self._guidance_scale1
    
    @property
    def guidance_scale2(self):
        return self._guidance_scale2

    # here `guidance_scale` is defined analog to the guidance weight `w` of equation (2)
    # of the Imagen paper: https://arxiv.org/pdf/2205.11487.pdf . `guidance_scale = 1`
    # corresponds to doing no classifier free guidance.
    @property
    def do_classifier_free_guidance(self):
        return True

    @property
    def num_timesteps(self):
        return self._num_timesteps

    @torch.no_grad()
    def __call__(
        self,
        ref_image: Union[PIL.Image.Image, List[PIL.Image.Image], torch.FloatTensor],
        transformed_images: Union[PIL.Image.Image, List[PIL.Image.Image], torch.FloatTensor],
        pose_cond_tensor: Union[PIL.Image.Image, List[PIL.Image.Image], torch.FloatTensor],
        prompts: Union[PIL.Image.Image, List[PIL.Image.Image], torch.FloatTensor],
        uncond_prompts: Union[PIL.Image.Image, List[PIL.Image.Image], torch.FloatTensor],
        height: int = 576,
        width: int = 1024,
        num_frames: Optional[int] = None,
        num_inference_steps: int = 25,
        min_guidance_scale1=1.0,
        max_guidance_scale1=3.0,
        min_guidance_scale2=1.0,
        max_guidance_scale2=3.0,
        fps: int = 7,
        motion_bucket_id: int = 127,
        noise_aug_strength: int = 0.02,
        decode_chunk_size: Optional[int] = None,
        num_videos_per_prompt: Optional[int] = 1,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        latents: Optional[torch.FloatTensor] = None,
        output_type: Optional[str] = "pil",
        callback_on_step_end: Optional[Callable[[int, int, Dict], None]] = None,
        callback_on_step_end_tensor_inputs: List[str] = ["latents"],
        return_dict: bool = True,
        overlap=7,
        shift_offset=0,
        frames_per_batch=14,
        i2i_noise_strength=1.0,
        arcface_embeddings=None,
        device = "cuda",
    ):
        r"""
        The call function to the pipeline for generation.

        Args:
            image (`PIL.Image.Image` or `List[PIL.Image.Image]` or `torch.FloatTensor`):
                Image or images to guide image generation. 
            height (`int`, *optional*, defaults to `self.unet.config.sample_size * self.vae_scale_factor`):
                The height in pixels of the generated image.
            width (`int`, *optional*, defaults to `self.unet.config.sample_size * self.vae_scale_factor`):
                The width in pixels of the generated image.
            num_frames (`int`, *optional*):
                The number of video frames to generate. Defaults to 14 for `stable-video-diffusion-img2vid` and to 25 for `stable-video-diffusion-img2vid-xt`
            num_inference_steps (`int`, *optional*, defaults to 25):
                The number of denoising steps. More denoising steps usually lead to a higher quality image at the
                expense of slower inference. This parameter is modulated by `strength`.
            min_guidance_scale (`float`, *optional*, defaults to 1.0):
                The minimum guidance scale. Used for the classifier free guidance with first frame.
            max_guidance_scale (`float`, *optional*, defaults to 3.0):
                The maximum guidance scale. Used for the classifier free guidance with last frame.
            fps (`int`, *optional*, defaults to 7):
                Frames per second. The rate at which the generated images shall be exported to a video after generation.
                Note that Stable Diffusion Video's UNet was micro-conditioned on fps-1 during training.
            motion_bucket_id (`int`, *optional*, defaults to 127):
                The motion bucket ID. Used as conditioning for the generation. The higher the number the more motion will be in the video.
            noise_aug_strength (`int`, *optional*, defaults to 0.02):
                The amount of noise added to the init image, the higher it is the less the video will look like the init image. Increase it for more motion.
            decode_chunk_size (`int`, *optional*):
                The number of frames to decode at a time. The higher the chunk size, the higher the temporal consistency
                between frames, but also the higher the memory consumption. By default, the decoder will decode all frames at once
                for maximal quality. Reduce `decode_chunk_size` to reduce memory usage.
            num_videos_per_prompt (`int`, *optional*, defaults to 1):
                The number of images to generate per prompt.
            generator (`torch.Generator` or `List[torch.Generator]`, *optional*):
                A [`torch.Generator`](https://pytorch.org/docs/stable/generated/torch.Generator.html) to make
                generation deterministic.
            latents (`torch.FloatTensor`, *optional*):
                Pre-generated noisy latents sampled from a Gaussian distribution, to be used as inputs for image
                generation. Can be used to tweak the same generation with different prompts. If not provided, a latents
                tensor is generated by sampling using the supplied random `generator`.
            output_type (`str`, *optional*, defaults to `"pil"`):
                The output format of the generated image. Choose between `PIL.Image` or `np.array`.
            callback_on_step_end (`Callable`, *optional*):
                A function that calls at the end of each denoising steps during the inference. The function is called
                with the following arguments: `callback_on_step_end(self: DiffusionPipeline, step: int, timestep: int,
                callback_kwargs: Dict)`. `callback_kwargs` will include a list of all tensors as specified by
                `callback_on_step_end_tensor_inputs`.
            callback_on_step_end_tensor_inputs (`List`, *optional*):
                The list of tensor inputs for the `callback_on_step_end` function. The tensors specified in the list
                will be passed as `callback_kwargs` argument. You will only be able to include variables listed in the
                `._callback_tensor_inputs` attribute of your pipeline class.
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether or not to return a [`~pipelines.stable_diffusion.StableDiffusionPipelineOutput`] instead of a
                plain tuple.

        Returns:
            [`~pipelines.stable_diffusion.StableVideoDiffusionPipelineOutput`] or `tuple`:
                If `return_dict` is `True`, [`~pipelines.stable_diffusion.StableVideoDiffusionPipelineOutput`] is returned,
                otherwise a `tuple` is returned where the first element is a list of list with the generated frames.

        Examples:

        ```py
        from diffusers import StableVideoDiffusionPipeline
        from diffusers.utils import load_image, export_to_video

        pipe = StableVideoDiffusionPipeline.from_pretrained("stabilityai/stable-video-diffusion-img2vid-xt", torch_dtype=torch.float16, variant="fp16")
        pipe.to("cuda")

        image = load_image("https://lh3.googleusercontent.com/y-iFOHfLTwkuQSUegpwDdgKmOjRSTvPxat63dQLB25xkTs4lhIbRUFeNBWZzYf370g=s1200")
        image = image.resize((1024, 576))

        frames = pipe(image, num_frames=25, decode_chunk_size=8).frames[0]
        export_to_video(frames, "generated.mp4", fps=7)
        ```
        """
        # 0. Default height and width to unet
        height = height or self.unet.config.sample_size * self.vae_scale_factor
        width = width or self.unet.config.sample_size * self.vae_scale_factor

        num_frames = num_frames if num_frames is not None else self.unet.config.num_frames
        decode_chunk_size = decode_chunk_size if decode_chunk_size is not None else num_frames

        # 1. Check inputs. Raise error if not correct
        self.check_inputs(ref_image, height, width)

        # 2. Define call parameters
        if isinstance(ref_image, PIL.Image.Image):
            batch_size = 1
        elif isinstance(ref_image, list):
            batch_size = len(ref_image)
        else:
            batch_size = ref_image.shape[0]

        # here `guidance_scale` is defined analog to the guidance weight `w` of equation (2)
        # of the Imagen paper: https://arxiv.org/pdf/2205.11487.pdf . `guidance_scale = 1`
        # corresponds to doing no classifier free guidance.
        do_classifier_free_guidance = True

        # 3. Prepare image embeds
        image_embeddings, prompts = self._encode_image(
            transformed_images, 
            arcface_embeddings,
            prompts,
            uncond_prompts,
            num_frames,
            device, 
            num_videos_per_prompt, 
            do_classifier_free_guidance
        )        

        # NOTE: Stable Diffusion Video was conditioned on fps - 1, which
        # is why it is reduced here.
        # See: https://github.com/Stability-AI/generative-models/blob/ed0997173f98eaf8f4edf7ba5fe8f15c6b877fd3/scripts/sampling/simple_video_sample.py#L188
        # fps = fps - 1

        # 4. Encode input image using VAE
        needs_upcasting = (self.vae.dtype == torch.float16 or self.vae.dtype == torch.bfloat16) and self.vae.config.force_upcast
        vae_dtype = self.vae.dtype
        if needs_upcasting:
            self.vae.to(dtype=torch.float32)
        
        # Prepare ref image latents
        ref_image_tensor = ref_image.to(
            dtype=self.vae.dtype, device=self.vae.device
        )
 
        ref_image_latents = self.vae.encode(ref_image_tensor).latent_dist.mean
        ref_image_latents = ref_image_latents * 0.18215  # (b, 4, h, w)

        noise = randn_tensor(
            ref_image_tensor.shape, 
            generator=generator, 
            device=self.vae.device, 
            dtype=self.vae.dtype)
        
        ref_image_tensor = ref_image_tensor + noise_aug_strength * noise

        image_latents = self._encode_vae_image(
            ref_image_tensor,
            device=device,
            num_videos_per_prompt=num_videos_per_prompt,
            do_classifier_free_guidance=do_classifier_free_guidance,
        )
        image_latents = image_latents.to(image_embeddings.dtype)
        ref_image_latents = ref_image_latents.to(image_embeddings.dtype)
        
        # cast back to fp16 if needed
        if needs_upcasting:
            self.vae.to(dtype=vae_dtype)
        
        # Repeat the image latents for each frame so we can concatenate them with the noise
        # image_latents [batch, channels, height, width] ->[batch, num_frames, channels, height, width]
        image_latents = image_latents.unsqueeze(1).repeat(1, num_frames, 1, 1, 1)        

        print('fps', fps)
        print('motion_bucket_id', motion_bucket_id)
        print('num_inference_steps', num_inference_steps)
        print('noise_aug_strength', noise_aug_strength)
        print('min_guidance_scale1', min_guidance_scale1)
        print('max_guidance_scale1', max_guidance_scale1)
        print('min_guidance_scale2', min_guidance_scale2)
        print('max_guidance_scale2', max_guidance_scale2)
        print('overlap', overlap)
        print('shift_offset', shift_offset)
        print('frames_per_batch', frames_per_batch)
        print('i2i_noise_strength', i2i_noise_strength)
        
         # 5. Get Added Time IDs
        added_time_ids = self._get_add_time_ids(
            fps,
            motion_bucket_id,
            noise_aug_strength,
            image_embeddings.dtype,
            batch_size,
            num_videos_per_prompt,
            do_classifier_free_guidance,
        )
        added_time_ids = added_time_ids.to(device, dtype=self.unet.dtype)

        # 4. Prepare timesteps
        self.scheduler.set_timesteps(num_inference_steps, device=device)
        timesteps, num_inference_steps = self.get_timesteps(num_inference_steps, i2i_noise_strength, device)
        latent_timestep = timesteps[:1].repeat(batch_size * num_videos_per_prompt)

        # 5. Prepare latent variables
        num_channels_latents = self.unet.config.in_channels
        latents = self.prepare_latents(
            batch_size * num_videos_per_prompt,
            num_frames,
            num_channels_latents,
            height,
            width,
            image_embeddings.dtype,
            device,
            generator,
            latents,
            ref_image_latents,
            timestep=latent_timestep
        )

        # Prepare a list of pose condition images
        pose_cond_tensor = pose_cond_tensor.to(
            device=device, dtype=self.pose_guider.dtype
        )
        face_mask = pose_cond_tensor[0, :1, :1]

        pose_fea = self.pose_guider(pose_cond_tensor).transpose(
            1, 2
        ) # (bs, f, c, H, W)
       
        # 7. Prepare guidance scale
        guidance_scale = torch.linspace(
            min_guidance_scale1, 
            max_guidance_scale1, 
            num_inference_steps)
        guidance_scale1 = guidance_scale.to(device, latents.dtype)

        guidance_scale = torch.linspace(
            min_guidance_scale2, 
            max_guidance_scale2, 
            num_inference_steps)
        guidance_scale2 = guidance_scale.to(device, latents.dtype)
        
        self._guidance_scale1 = guidance_scale1
        self._guidance_scale2 = guidance_scale2

        # 8. Denoising loop
        latents_all = latents # for any-frame generation

        num_warmup_steps = len(timesteps) - num_inference_steps * self.scheduler.order
        self._num_timesteps = len(timesteps)
        shift = 0
        with self.progress_bar(total=num_inference_steps) as progress_bar:
            for i, t in enumerate(timesteps):

                # init 
                pred_latents = torch.zeros_like(
                    latents_all,
                    dtype=self.unet.dtype,
                )
                counter = torch.zeros(
                    (latents_all.shape[0], num_frames, 1, 1, 1),
                    dtype=self.unet.dtype,
                ).to(device=latents_all.device)

                for batch, index_start in enumerate(range(0, num_frames, frames_per_batch - overlap)):
                    self.scheduler._step_index = None
                    index_start -= shift
                    def indice_slice(tensor, idx_list):
                        tensor_list = []
                        for idx in idx_list:
                            idx = idx % tensor.shape[1]
                            tensor_list.append(tensor[:,idx])
                        return torch.stack(tensor_list, 1)
                    idx_list = list(range(index_start, index_start+frames_per_batch))
                    latents = indice_slice(latents_all, idx_list)
                    pose_cond_fea = indice_slice(pose_fea, idx_list)
                    image_latents_input = indice_slice(image_latents, idx_list)
                    batch_image_embeddings = indice_slice(image_embeddings, idx_list)
                    batch_prompts = indice_slice(prompts, idx_list)

                    cross_attention_kwargs = {'ip_adapter_masks': [face_mask]}
                    latent_model_input = torch.cat([latents] * 3) if do_classifier_free_guidance else latents
                    latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)
                    pose_cond_fea = pose_cond_fea.repeat(3 if do_classifier_free_guidance else 1, 1, 1, 1, 1)

                    # Concatenate image_latents over channels dimention
                    latent_model_input = torch.cat([
                        latent_model_input, 
                        image_latents_input], dim=2)
                    # predict the noise residual
                    noise_pred = self.unet(
                        latent_model_input,
                        t,
                        encoder_hidden_states=(batch_image_embeddings.flatten(0,1), [batch_prompts.flatten(0,1)]),
                        pose_cond_fea=pose_cond_fea,
                        added_time_ids=added_time_ids,
                        return_dict=False,
                    )[0]        
                    # perform guidance
                    if do_classifier_free_guidance:
                        noise_pred_uncond, noise_pred_drop, noise_pred_cond = noise_pred.chunk(3)
                        noise_pred = noise_pred_uncond + self.guidance_scale1[i] * (noise_pred_drop - noise_pred_uncond) + self.guidance_scale2[i] * (noise_pred_cond - noise_pred_drop)

                    # compute the previous noisy sample x_t -> x_t-1
                    latents = self.scheduler.step(noise_pred, t.to(self.unet.dtype), latents, noise_type='random').prev_sample

                    if callback_on_step_end is not None:
                        callback_kwargs = {}
                        for k in callback_on_step_end_tensor_inputs:
                            callback_kwargs[k] = locals()[k]
                        callback_outputs = callback_on_step_end(self, i, t, callback_kwargs)

                        latents = callback_outputs.pop("latents", latents)

                    for f_idx in range(frames_per_batch):
                        p = (index_start + f_idx) % pred_latents.shape[1]
                        pred_latents[:, p] += latents[:, f_idx]
                        counter[:, p] += 1
                shift += shift_offset
                shift = shift % frames_per_batch

                pred_latents  = pred_latents / counter
                latents_all = pred_latents

                if i == len(timesteps) - 1 or ((i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0):
                    progress_bar.update()

        latents = latents_all
        if not output_type == "latent":
            # cast back to fp16 if needed
            if needs_upcasting:
                self.vae.to(dtype=vae_dtype)
            frames = self.decode_latents(latents, num_frames, decode_chunk_size)
        else:
            frames = latents

        self.maybe_free_model_hooks()

        if not return_dict:
            return frames
        return HunyuanSVDPipelineOutput(frames=frames)


def _resize_with_antialiasing(input, size, interpolation="bicubic", align_corners=True):
    h, w = input.shape[-2:]
    factors = (h / size[0], w / size[1])

    # First, we have to determine sigma
    # Taken from skimage: https://github.com/scikit-image/scikit-image/blob/v0.19.2/skimage/transform/_warps.py#L171
    sigmas = (
        max((factors[0] - 1.0) / 2.0, 0.001),
        max((factors[1] - 1.0) / 2.0, 0.001),
    )

    # Now kernel size. Good results are for 3 sigma, but that is kind of slow. Pillow uses 1 sigma
    # https://github.com/python-pillow/Pillow/blob/master/src/libImaging/Resample.c#L206
    # But they do it in the 2 passes, which gives better results. Let's try 2 sigmas for now
    ks = int(max(2.0 * 2 * sigmas[0], 3)), int(max(2.0 * 2 * sigmas[1], 3))

    # Make sure it is odd
    if (ks[0] % 2) == 0:
        ks = ks[0] + 1, ks[1]

    if (ks[1] % 2) == 0:
        ks = ks[0], ks[1] + 1

    input = _gaussian_blur2d(input, ks, sigmas)

    output = torch.nn.functional.interpolate(input, size=size, mode=interpolation, align_corners=align_corners)
    return output


def _compute_padding(kernel_size):
    """Compute padding tuple."""
    # 4 or 6 ints:  (padding_left, padding_right,padding_top,padding_bottom)
    # https://pytorch.org/docs/stable/nn.html#torch.nn.functional.pad
    if len(kernel_size) < 2:
        raise AssertionError(kernel_size)
    computed = [k - 1 for k in kernel_size]

    # for even kernels we need to do asymmetric padding :(
    out_padding = 2 * len(kernel_size) * [0]

    for i in range(len(kernel_size)):
        computed_tmp = computed[-(i + 1)]

        pad_front = computed_tmp // 2
        pad_rear = computed_tmp - pad_front

        out_padding[2 * i + 0] = pad_front
        out_padding[2 * i + 1] = pad_rear

    return out_padding


def _filter2d(input, kernel):
    # prepare kernel
    b, c, h, w = input.shape
    tmp_kernel = kernel[:, None, ...].to(device=input.device, dtype=input.dtype)

    tmp_kernel = tmp_kernel.expand(-1, c, -1, -1)

    height, width = tmp_kernel.shape[-2:]

    padding_shape: list[int] = _compute_padding([height, width])
    input = torch.nn.functional.pad(input, padding_shape, mode="reflect")

    # kernel and input tensor reshape to align element-wise or batch-wise params
    tmp_kernel = tmp_kernel.reshape(-1, 1, height, width)
    input = input.view(-1, tmp_kernel.size(0), input.size(-2), input.size(-1))

    # convolve the tensor with the kernel.
    output = torch.nn.functional.conv2d(input, tmp_kernel, groups=tmp_kernel.size(0), padding=0, stride=1)

    out = output.view(b, c, h, w)
    return out


def _gaussian(window_size: int, sigma):
    if isinstance(sigma, float):
        sigma = torch.tensor([[sigma]])

    batch_size = sigma.shape[0]

    x = (torch.arange(window_size, device=sigma.device, dtype=sigma.dtype) - window_size // 2).expand(batch_size, -1)

    if window_size % 2 == 0:
        x = x + 0.5

    gauss = torch.exp(-x.pow(2.0) / (2 * sigma.pow(2.0)))

    return gauss / gauss.sum(-1, keepdim=True)


def _gaussian_blur2d(input, kernel_size, sigma):
    if isinstance(sigma, tuple):
        sigma = torch.tensor([sigma], dtype=input.dtype)
    else:
        sigma = sigma.to(dtype=input.dtype)

    ky, kx = int(kernel_size[0]), int(kernel_size[1])
    bs = sigma.shape[0]
    kernel_x = _gaussian(kx, sigma[:, 1].view(bs, 1))
    kernel_y = _gaussian(ky, sigma[:, 0].view(bs, 1))
    out_x = _filter2d(input, kernel_x[..., None, :])
    out = _filter2d(out_x, kernel_y[..., None])

    return out