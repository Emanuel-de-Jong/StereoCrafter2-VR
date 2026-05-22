import os
import math
import torch
import torch.nn.functional as F
import numpy as np
from diffusers.utils import export_to_video
from PIL import Image
from decord import VideoReader, cpu
from diffusers import WanVACETransformer3DModel, AutoencoderKLWan
from diffusers.video_processor import VideoProcessor
from transformers import AutoTokenizer, UMT5EncoderModel
import ftfy
import html
import re
from fire import Fire


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DTYPE = torch.bfloat16
PROMPT = ""


class FlowMatchScheduler():

    def __init__(self,):
        self.set_timesteps_fn = FlowMatchScheduler.set_timesteps_wan
        self.num_train_timesteps = 1000

    @staticmethod
    def set_timesteps_wan(num_inference_steps=100, denoising_strength=1.0, shift=None):
        sigma_min = 0.0
        sigma_max = 1.0
        shift = 5 if shift is None else shift
        num_train_timesteps = 1000
        sigma_start = sigma_min + (sigma_max - sigma_min) * denoising_strength
        sigmas = torch.linspace(sigma_start, sigma_min, num_inference_steps + 1)[:-1]
        sigmas = shift * sigmas / (1 + (shift - 1) * sigmas)
        timesteps = sigmas * num_train_timesteps
        return sigmas, timesteps
        
    def set_timesteps(self, num_inference_steps=100, denoising_strength=1.0, **kwargs):
        self.sigmas, self.timesteps = self.set_timesteps_fn(
            num_inference_steps=num_inference_steps,
            denoising_strength=denoising_strength,
            **kwargs,
        )

    def step(self, model_output, timestep, sample, to_final=False, **kwargs):
        if isinstance(timestep, torch.Tensor):
            timestep = timestep.cpu()
        timestep_id = torch.argmin((self.timesteps - timestep).abs())
        sigma = self.sigmas[timestep_id]
        if to_final or timestep_id + 1 >= len(self.timesteps):
            sigma_ = 0
        else:
            sigma_ = self.sigmas[timestep_id + 1]
        prev_sample = sample + model_output * (sigma_ - sigma)
        return prev_sample


def basic_clean(text):
    text = ftfy.fix_text(text)
    text = html.unescape(html.unescape(text))
    return text.strip()

def whitespace_clean(text):
    text = re.sub(r"\s+", " ", text)
    text = text.strip()
    return text

def prompt_clean(text):
    text = whitespace_clean(basic_clean(text))
    return text


def get_t5_prompt_embeds(
    prompt = None,
    num_videos_per_prompt = 1,
    max_sequence_length = 226,
    device = None,
    dtype = None,
    tokenizer = None,
    text_encoder = None,
):
    # device = device or self._execution_device
    # dtype = dtype or self.text_encoder.dtype

    prompt = [prompt] if isinstance(prompt, str) else prompt
    prompt = [prompt_clean(u) for u in prompt]
    batch_size = len(prompt)

    text_inputs = tokenizer(
        prompt,
        padding="max_length",
        max_length=max_sequence_length,
        truncation=True,
        add_special_tokens=True,
        return_attention_mask=True,
        return_tensors="pt",
    )
    text_input_ids, mask = text_inputs.input_ids, text_inputs.attention_mask
    seq_lens = mask.gt(0).sum(dim=1).long()

    prompt_embeds = text_encoder(text_input_ids.to(device), mask.to(device)).last_hidden_state
    prompt_embeds = prompt_embeds.to(dtype=dtype, device=device)
    prompt_embeds = [u[:v] for u, v in zip(prompt_embeds, seq_lens)]
    prompt_embeds = torch.stack(
        [torch.cat([u, u.new_zeros(max_sequence_length - u.size(0), u.size(1))]) for u in prompt_embeds], dim=0
    )

    # duplicate text embeddings for each generation per prompt, using mps friendly method
    _, seq_len, _ = prompt_embeds.shape
    prompt_embeds = prompt_embeds.repeat(1, num_videos_per_prompt, 1)
    prompt_embeds = prompt_embeds.view(batch_size * num_videos_per_prompt, seq_len, -1)

    return prompt_embeds


def encode_prompt(
    prompt,
    negative_prompt = None,
    do_classifier_free_guidance = True,
    num_videos_per_prompt = 1,
    prompt_embeds = None,
    negative_prompt_embeds = None,
    max_sequence_length = 226,
    device = None,
    dtype = None,
    tokenizer = None,
    text_encoder = None,
):
    r"""
    Encodes the prompt into text encoder hidden states.

    Args:
        prompt (`str` or `List[str]`, *optional*):
            prompt to be encoded
        negative_prompt (`str` or `List[str]`, *optional*):
            The prompt or prompts not to guide the image generation. If not defined, one has to pass
            `negative_prompt_embeds` instead. Ignored when not using guidance (i.e., ignored if `guidance_scale` is
            less than `1`).
        do_classifier_free_guidance (`bool`, *optional*, defaults to `True`):
            Whether to use classifier free guidance or not.
        num_videos_per_prompt (`int`, *optional*, defaults to 1):
            Number of videos that should be generated per prompt. torch device to place the resulting embeddings on
        prompt_embeds (`torch.Tensor`, *optional*):
            Pre-generated text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt weighting. If not
            provided, text embeddings will be generated from `prompt` input argument.
        negative_prompt_embeds (`torch.Tensor`, *optional*):
            Pre-generated negative text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt
            weighting. If not provided, negative_prompt_embeds will be generated from `negative_prompt` input
            argument.
        device: (`torch.device`, *optional*):
            torch device
        dtype: (`torch.dtype`, *optional*):
            torch dtype
    """
    # device = device or self._execution_device

    prompt = [prompt] if isinstance(prompt, str) else prompt
    if prompt is not None:
        batch_size = len(prompt)
    else:
        batch_size = prompt_embeds.shape[0]

    if prompt_embeds is None:
        prompt_embeds = get_t5_prompt_embeds(
            prompt=prompt,
            num_videos_per_prompt=num_videos_per_prompt,
            max_sequence_length=max_sequence_length,
            device=device,
            dtype=dtype,
            tokenizer=tokenizer,
            text_encoder=text_encoder,
        )

    if do_classifier_free_guidance and negative_prompt_embeds is None:
        negative_prompt = negative_prompt or ""
        negative_prompt = batch_size * [negative_prompt] if isinstance(negative_prompt, str) else negative_prompt

        if prompt is not None and type(prompt) is not type(negative_prompt):
            raise TypeError(
                f"`negative_prompt` should be the same type to `prompt`, but got {type(negative_prompt)} !="
                f" {type(prompt)}."
            )
        elif batch_size != len(negative_prompt):
            raise ValueError(
                f"`negative_prompt`: {negative_prompt} has batch size {len(negative_prompt)}, but `prompt`:"
                f" {prompt} has batch size {batch_size}. Please make sure that passed `negative_prompt` matches"
                " the batch size of `prompt`."
            )

        negative_prompt_embeds = get_t5_prompt_embeds(
            prompt=negative_prompt,
            num_videos_per_prompt=num_videos_per_prompt,
            max_sequence_length=max_sequence_length,
            device=device,
            dtype=dtype,
            tokenizer=tokenizer,
            text_encoder=text_encoder,
        )

    return prompt_embeds, negative_prompt_embeds


def prepare_masks(
    mask: torch.Tensor,
    reference_images = None,
    # generator = None,
    transformer_patch_size = None,
    vae_scale_factor_temporal = None,
    vae_scale_factor_spatial = None,
) -> torch.Tensor:
    # if isinstance(generator, list):
    #     # TODO: support this
    #     raise ValueError("Passing a list of generators is not yet supported. This may be supported in the future.")

    if reference_images is None:
        # For each batch of video, we set no reference image (as one or more can be passed by user)
        reference_images = [[None] for _ in range(mask.shape[0])]
    else:
        if mask.shape[0] != len(reference_images):
            raise ValueError(
                f"Batch size of `mask` {mask.shape[0]} and length of `reference_images` {len(reference_images)} does not match."
            )

    # if mask.shape[0] != 1:
    #     # TODO: support this
    #     raise ValueError(
    #         "Generating with more than one video is not yet supported. This may be supported in the future."
    #     )

    # transformer_patch_size = (
    #     self.transformer.config.patch_size[1]
    #     if self.transformer is not None
    #     else self.transformer_2.config.patch_size[1]
    # )

    mask_list = []
    for mask_, reference_images_batch in zip(mask, reference_images):
        num_channels, num_frames, height, width = mask_.shape
        new_num_frames = (num_frames + vae_scale_factor_temporal - 1) // vae_scale_factor_temporal
        new_height = height // (vae_scale_factor_spatial * transformer_patch_size) * transformer_patch_size
        new_width = width // (vae_scale_factor_spatial * transformer_patch_size) * transformer_patch_size
        mask_ = mask_[0, :, :, :]
        mask_ = mask_.view(
            num_frames, new_height, vae_scale_factor_spatial, new_width, vae_scale_factor_spatial
        )
        mask_ = mask_.permute(2, 4, 0, 1, 3).flatten(0, 1)  # [8x8, num_frames, new_height, new_width]
        mask_ = torch.nn.functional.interpolate(
            mask_.unsqueeze(0), size=(new_num_frames, new_height, new_width), mode="nearest-exact"
        ).squeeze(0)
        num_ref_images = len(reference_images_batch)
        if num_ref_images > 0:
            mask_padding = torch.zeros_like(mask_[:, :num_ref_images, :, :])
            mask_ = torch.cat([mask_padding, mask_], dim=1)
        mask_list.append(mask_)
    return torch.stack(mask_list)


def preprocess_conditions(
    video = None,
    mask = None,
    reference_images = None,
    batch_size: int = 1,
    height: int = 480,
    width: int = 832,
    num_frames: int = 81,
    dtype = None,
    device = None,
    video_processor = None,
    base = None,
):
    if video is not None:
        # base = self.vae_scale_factor_spatial * (
        #     self.transformer.config.patch_size[1]
        #     if self.transformer is not None
        #     else self.transformer_2.config.patch_size[1]
        # )
        video_height, video_width = video_processor.get_default_height_width(video[0])

        if video_height * video_width > height * width:
            scale = min(width / video_width, height / video_height)
            video_height, video_width = int(video_height * scale), int(video_width * scale)

        if video_height % base != 0 or video_width % base != 0:
            # logger.warning(
            #     f"Video height and width should be divisible by {base}, but got {video_height} and {video_width}. "
            # )
            video_height = (video_height // base) * base
            video_width = (video_width // base) * base

        assert video_height * video_width <= height * width

        video = video_processor.preprocess_video(video, video_height, video_width)
        image_size = (video_height, video_width)  # Use the height/width of video (with possible rescaling)
    else:
        video = torch.zeros(batch_size, 3, num_frames, height, width, dtype=dtype, device=device)
        image_size = (height, width)  # Use the height/width provider by user


    if mask is not None:
        mask = video_processor.preprocess_video(mask, image_size[0], image_size[1])
        mask = torch.clamp((mask + 1) / 2, min=0, max=1)
    else:
        mask = torch.ones_like(video)

    video = video.to(dtype=dtype, device=device)
    mask = mask.to(dtype=dtype, device=device)

    # Make a list of list of images where the outer list corresponds to video batch size and the inner list
    # corresponds to list of conditioning images per video
    if reference_images is None or isinstance(reference_images, Image.Image):
        reference_images = [[reference_images] for _ in range(video.shape[0])]
    elif isinstance(reference_images, (list, tuple)) and isinstance(next(iter(reference_images)), Image.Image):
        reference_images = [reference_images]
    elif (
        isinstance(reference_images, (list, tuple))
        and isinstance(next(iter(reference_images)), list)
        and isinstance(next(iter(reference_images[0])), Image.Image)
    ):
        reference_images = reference_images
    else:
        raise ValueError(
            "`reference_images` has to be of type `PIL.Image.Image` or `list` of `PIL.Image.Image`, or "
            "`list` of `list` of `PIL.Image.Image`, but is {type(reference_images)}"
        )

    if video.shape[0] != len(reference_images):
        raise ValueError(
            f"Batch size of `video` {video.shape[0]} and length of `reference_images` {len(reference_images)} does not match."
        )

    ref_images_lengths = [len(reference_images_batch) for reference_images_batch in reference_images]
    if any(l != ref_images_lengths[0] for l in ref_images_lengths):
        raise ValueError(
            f"All batches of `reference_images` should have the same length, but got {ref_images_lengths}. Support for this "
            "may be added in the future."
        )

    reference_images_preprocessed = []
    for i, reference_images_batch in enumerate(reference_images):
        preprocessed_images = []
        for j, image in enumerate(reference_images_batch):
            if image is None:
                continue
            image = video_processor.preprocess(image, None, None)
            img_height, img_width = image.shape[-2:]
            scale = min(image_size[0] / img_height, image_size[1] / img_width)
            new_height, new_width = int(img_height * scale), int(img_width * scale)
            resized_image = torch.nn.functional.interpolate(
                image, size=(new_height, new_width), mode="bilinear", align_corners=False
            ).squeeze(0)  # [C, H, W]
            top = (image_size[0] - new_height) // 2
            left = (image_size[1] - new_width) // 2
            canvas = torch.ones(3, *image_size, device=device, dtype=dtype)
            canvas[:, top : top + new_height, left : left + new_width] = resized_image
            preprocessed_images.append(canvas)
        reference_images_preprocessed.append(preprocessed_images)

    return video, mask, reference_images_preprocessed


def prepare_video_latents(
    video: torch.Tensor,
    mask: torch.Tensor,
    reference_images = None,
    device = None,
    vae = None,
) -> torch.Tensor:
    # device = device or self._execution_device

    # if isinstance(generator, list):
    #     # TODO: support this
    #     raise ValueError("Passing a list of generators is not yet supported. This may be supported in the future.")

    if reference_images is None:
        # For each batch of video, we set no re
        # ference image (as one or more can be passed by user)
        reference_images = [[None] for _ in range(video.shape[0])]
    else:
        if video.shape[0] != len(reference_images):
            raise ValueError(
                f"Batch size of `video` {video.shape[0]} and length of `reference_images` {len(reference_images)} does not match."
            )

    # if video.shape[0] != 1:
    #     # TODO: support this
    #     raise ValueError(
    #         "Generating with more than one video is not yet supported. This may be supported in the future."
    #     )

    vae_dtype = vae.dtype
    video = video.to(dtype=vae_dtype)

    latents_mean = torch.tensor(vae.config.latents_mean, device=device, dtype=torch.float32).view(
        1, vae.config.z_dim, 1, 1, 1
    )
    latents_std = 1.0 / torch.tensor(vae.config.latents_std, device=device, dtype=torch.float32).view(
        1, vae.config.z_dim, 1, 1, 1
    )

    if mask is None:
        # latents = retrieve_latents(vae.encode(video), generator, sample_mode="argmax").unbind(0)
        latents = vae.encode(video).latent_dist.sample()
        latents = ((latents.float() - latents_mean) * latents_std).to(vae_dtype)
    else:
        mask = torch.where(mask > 0.5, 1.0, 0.0).to(dtype=vae_dtype)
        inactive = video * (1 - mask)
        reactive = video * mask
        # inactive = retrieve_latents(vae.encode(inactive), generator, sample_mode="argmax")
        inactive = vae.encode(inactive).latent_dist.sample()
        # reactive = retrieve_latents(vae.encode(reactive), generator, sample_mode="argmax")
        reactive = vae.encode(reactive).latent_dist.sample()
        inactive = ((inactive.float() - latents_mean) * latents_std).to(vae_dtype)
        reactive = ((reactive.float() - latents_mean) * latents_std).to(vae_dtype)
        latents = torch.cat([inactive, reactive], dim=1)


    latent_list = []
    for latent, reference_images_batch in zip(latents, reference_images):
        for reference_image in reference_images_batch:
            assert reference_image.ndim == 3
            reference_image = reference_image.to(dtype=vae_dtype)
            reference_image = reference_image[None, :, None, :, :]  # [1, C, 1, H, W]
            # reference_latent = retrieve_latents(vae.encode(reference_image), generator, sample_mode="argmax")
            reference_latent = vae.encode(reference_image).latent_dist.sample()
            reference_latent = ((reference_latent.float() - latents_mean) * latents_std).to(vae_dtype)
            reference_latent = reference_latent.squeeze(0)  # [C, 1, H, W]
            reference_latent = torch.cat([reference_latent, torch.zeros_like(reference_latent)], dim=0)
            latent = torch.cat([reference_latent.squeeze(0), latent], dim=1)
        latent_list.append(latent)

    return torch.stack(latent_list)


def blend_h(a: torch.Tensor, b: torch.Tensor, overlap_size: int) -> torch.Tensor:
    """水平方向融合 Latents，支持 [B, C, F, H, W]"""
    weight_b = (torch.arange(overlap_size).view(1, 1, 1, 1, -1) / overlap_size).to(b.device, dtype=b.dtype)
    b[:, :, :, :, :overlap_size] = (1 - weight_b) * a[:, :, :, :, -overlap_size:] + weight_b * b[:, :, :, :, :overlap_size]
    return b


def blend_v(a: torch.Tensor, b: torch.Tensor, overlap_size: int) -> torch.Tensor:
    """垂直方向融合 Latents，支持 [B, C, F, H, W]"""
    weight_b = (torch.arange(overlap_size).view(1, 1, 1, -1, 1) / overlap_size).to(b.device, dtype=b.dtype)
    b[:, :, :, :overlap_size, :] = (1 - weight_b) * a[:, :, :, -overlap_size:, :] + weight_b * b[:, :, :, :overlap_size, :]
    return b


def run_wan_pipeline(
    cond_frames, 
    mask_frames, 
    prompt_embeds, 
    transformer, 
    vae, 
    noise_scheduler, 
    videoprocessor,
    vae_scale_factor_spatial,
    vae_scale_factor_temporal,
    transformer_patch_size
):
    """封装单次 Wan 去噪 Pipeline 以供分块调用"""
    # 此时进入的 cond_frames 是正确的 [B, C, F, H, W]
    height, width = cond_frames.shape[3], cond_frames.shape[4]
    num_frames = cond_frames.shape[2]

    with torch.no_grad():
        # VideoProcessor 强制要求输入格式为 [B, F, C, H, W]
        # 所以我们在这里做一次临时的维度翻转：[B, C, F, H, W] -> [B, F, C, H, W]
        cond_frames_vp = cond_frames.permute(0, 2, 1, 3, 4)
        mask_frames_vp = mask_frames.permute(0, 2, 1, 3, 4)
        
        condition_video, mask, reference_images = preprocess_conditions(
            video=cond_frames_vp,
            mask=mask_frames_vp,
            reference_images=None,
            batch_size=1,
            height=height,
            width=width,
            num_frames=num_frames,
            dtype=DTYPE,
            device=DEVICE,
            video_processor=videoprocessor,
            base=vae_scale_factor_spatial * transformer_patch_size,
        )
    
        conditioning_latents = prepare_video_latents(condition_video, mask, reference_images, DEVICE, vae)
        mask_for_transformer = prepare_masks(mask, reference_images, transformer_patch_size, vae_scale_factor_temporal, vae_scale_factor_spatial).to(DEVICE, dtype=DTYPE)
        control_hidden_states = torch.cat([conditioning_latents, mask_for_transformer], dim=1).to(DTYPE)

    c = transformer.config.in_channels
    f = (num_frames - 1) // vae_scale_factor_temporal + 1
    h = height // vae_scale_factor_spatial
    w = width // vae_scale_factor_spatial
    
    latents = torch.randn(1, c, f, h, w, device=DEVICE, dtype=DTYPE)

    for i, t in enumerate(noise_scheduler.timesteps):
        timestep_tensor = t.unsqueeze(0).to(DEVICE, dtype=DTYPE)
        with torch.no_grad():
            model_pred = transformer(
                hidden_states=latents,
                timestep=timestep_tensor,
                encoder_hidden_states=prompt_embeds,
                control_hidden_states=control_hidden_states,
                return_dict=False,
            )[0]
        latents = noise_scheduler.step(model_pred, t, latents)
    
    return latents


def spatial_tiled_process(
    cond_frames, mask_frames, tile_num, tile_overlap, prompt_embeds, transformer, vae, noise_scheduler, videoprocessor,
    vae_scale_factor_spatial, vae_scale_factor_temporal, transformer_patch_size
):
    """处理单段视频的空间分块"""
    if tile_num == 1:
        return run_wan_pipeline(cond_frames, mask_frames, prompt_embeds, transformer, vae, noise_scheduler, videoprocessor, vae_scale_factor_spatial, vae_scale_factor_temporal, transformer_patch_size)

    height = cond_frames.shape[3]
    width = cond_frames.shape[4]
    
    # 确保切块大小能够被 VAE 和 Transformer Patch 的乘积（通常是 16）整除
    base = vae_scale_factor_spatial * transformer_patch_size
    tile_size = (
        int((height + tile_overlap * (tile_num - 1)) / tile_num) // base * base,
        int((width + tile_overlap * (tile_num - 1)) / tile_num) // base * base
    )
    tile_stride = (tile_size[0] - tile_overlap, tile_size[1] - tile_overlap)

    cols = []
    for i in range(tile_num):
        rows = []
        for j in range(tile_num):
            h_start = min(i * tile_stride[0], height - tile_size[0])
            w_start = min(j * tile_stride[1], width - tile_size[1])
            
            cond_tile = cond_frames[:, :, :, h_start : h_start + tile_size[0], w_start : w_start + tile_size[1]]
            mask_tile = mask_frames[:, :, :, h_start : h_start + tile_size[0], w_start : w_start + tile_size[1]]

            tile_latent = run_wan_pipeline(
                cond_tile, mask_tile, prompt_embeds, transformer, vae, noise_scheduler, videoprocessor,
                vae_scale_factor_spatial, vae_scale_factor_temporal, transformer_patch_size
            )
            rows.append(tile_latent)
        cols.append(rows)

    # 映射回 Latent 空间的 stride 和 overlap
    latent_stride = (tile_stride[0] // vae_scale_factor_spatial, tile_stride[1] // vae_scale_factor_spatial)
    latent_overlap = (tile_overlap // vae_scale_factor_spatial, tile_overlap // vae_scale_factor_spatial)

    # 融合 Latents
    results_cols = []
    for i, rows in enumerate(cols):
        results_rows = []
        for j, tile in enumerate(rows):
            if i > 0:
                tile = blend_v(cols[i - 1][j], tile, latent_overlap[0])
            if j > 0:
                tile = blend_h(rows[j - 1], tile, latent_overlap[1])
            results_rows.append(tile)
        results_cols.append(results_rows)

    pixels = []
    for i, rows in enumerate(results_cols):
        for j, tile in enumerate(rows):
            if i < len(results_cols) - 1:
                tile = tile[:, :, :, :latent_stride[0], :]
            if j < len(rows) - 1:
                tile = tile[:, :, :, :, :latent_stride[1]]
            rows[j] = tile
        pixels.append(torch.cat(rows, dim=4))
    
    return torch.cat(pixels, dim=3)


def main(
    pre_trained_path,
    transformer_path,
    input_video_path,
    save_dir,
    frames_chunk=79,
    frames_overlap=3,
    tile_overlap=128,
    tile_num=2,
    inference_steps=8,
):

    tokenizer = AutoTokenizer.from_pretrained(pre_trained_path, subfolder="tokenizer")
    text_encoder = UMT5EncoderModel.from_pretrained(pre_trained_path, subfolder="text_encoder", torch_dtype=DTYPE).to(DEVICE)
    vae = AutoencoderKLWan.from_pretrained(pre_trained_path, subfolder="vae", torch_dtype=DTYPE).to(DEVICE)
    transformer = WanVACETransformer3DModel.from_pretrained(transformer_path, torch_dtype=DTYPE).to(DEVICE)

    transformer.eval()
    vae.eval()
    text_encoder.eval()

    text_encoder.requires_grad_(False)
    vae.requires_grad_(False)
    transformer.requires_grad_(False)

    videoprocessor = VideoProcessor(vae_scale_factor=vae.config.scale_factor_spatial)
    transformer_patch_size = transformer.config.patch_size[1]
    vae_scale_factor_temporal = 2 ** sum(vae.temperal_downsample)
    vae_scale_factor_spatial = 2 ** len(vae.temperal_downsample)


    os.makedirs(save_dir, exist_ok=True)
    video_name = input_video_path.split("/")[-1].replace(".mp4", "").replace("_splatting_results", "") + "_inpainting_results"

    print("Encoding prompt...")
    with torch.no_grad():
        prompt_embeds, _ = encode_prompt(
            [PROMPT], do_classifier_free_guidance=False, max_sequence_length=226,
            device=DEVICE, dtype=DTYPE, tokenizer=tokenizer, text_encoder=text_encoder
        )

    print("Loading video...")
    video_reader = VideoReader(input_video_path, ctx=cpu(0))
    fps = video_reader.get_avg_fps()
    total_frames = len(video_reader)
    frame_indices = list(range(total_frames))
    frames = video_reader.get_batch(frame_indices)
    

    # [t,h,w,c] -> [1,c,t,h,w]
    frames = torch.from_numpy(frames.asnumpy()).permute(3, 0, 1, 2).unsqueeze(0).float() / 255.0

    height, width = frames.shape[3] // 2, frames.shape[4] // 2
    frames_left = frames[:, :, :, :height, :width]
    all_masks = frames[:, :, :, height:, :width]
    all_frames = frames[:, :, :, height:, width:]

    all_frames = all_frames * (1.0 - all_masks) + 0.5 * all_masks

    base = vae_scale_factor_spatial * transformer_patch_size
    h_orig, w_orig = all_frames.shape[3], all_frames.shape[4]
    
    # 1. 向上取整 (math.ceil)，倒推计算出能被 Tiling 完美拼接的“目标分辨率”
    min_tile_h = (h_orig + tile_overlap * (tile_num - 1)) / tile_num
    tile_size_h = math.ceil(min_tile_h / base) * base
    
    min_tile_w = (w_orig + tile_overlap * (tile_num - 1)) / tile_num
    tile_size_w = math.ceil(min_tile_w / base) * base
    
    tile_stride_h = tile_size_h - tile_overlap
    tile_stride_w = tile_size_w - tile_overlap
    
    target_h = tile_stride_h * (tile_num - 1) + tile_size_h
    target_w = tile_stride_w * (tile_num - 1) + tile_size_w

    # 2. 计算需要补充的边缘像素数
    pad_h = target_h - h_orig
    pad_w = target_w - w_orig

    if pad_h > 0 or pad_w > 0:
        print(f"Padding resolution from {w_orig}x{h_orig} to {target_w}x{target_h} to perfectly match Tiling output.")
        
        # 1. 临时取出 Batch 并对调通道和帧数维度: [1, C, F, H, W] -> [F, C, H, W] (变成标准的 4D 图片格式)
        frames_4d = all_frames[0].permute(1, 0, 2, 3) 
        
        # 2. 对 4D 张量进行边缘复制填充 (PyTorch 对此支持极其完美)
        frames_4d = F.pad(frames_4d, (0, pad_w, 0, pad_h), mode='replicate')
        
        # 3. 还原回 5D 视频张量: [F, C, H_new, W_new] -> [1, C, F, H_new, W_new]
        all_frames = frames_4d.permute(1, 0, 2, 3).unsqueeze(0)
        
        # Mask 填充 0 (constant 模式原生支持 5D，直接 pad 即可)
        all_masks = F.pad(all_masks, (0, pad_w, 0, pad_h), mode='constant', value=0)

    noise_scheduler = FlowMatchScheduler()
    noise_scheduler.set_timesteps(num_inference_steps=inference_steps, denoising_strength=1.0)

    generated_video_chunks = []
    
    print(f"Starting Temporal Chunking inference (Total Frames: {total_frames})...")
    
    # 记录全局已经完美生成的有效帧数
    global_len = 0
    
    while global_len < total_frames:
        if global_len == 0:
            # 第一段：从 0 开始
            cur_i = 0
            cur_chunk_size = min(frames_chunk, total_frames)
            # 安全修正（防止输入视频本身不到 81 帧）
            valid_chunk_size = ((cur_chunk_size - 1) // vae_scale_factor_temporal) * vae_scale_factor_temporal + 1
        else:
            # 正常步长推进
            cur_i = global_len - frames_overlap
            
            # 如果按正常步长取，超出了总帧数，说明这是最后一段
            if cur_i + frames_chunk > total_frames:
                # 强制把截取起点向前推，确保这一段刚好能取满 frames_chunk (81帧) 并直达视频结尾
                cur_i = max(0, total_frames - frames_chunk)
                
            cur_chunk_size = min(frames_chunk, total_frames - cur_i)
            valid_chunk_size = ((cur_chunk_size - 1) // vae_scale_factor_temporal) * vae_scale_factor_temporal + 1

        chunk_cond = all_frames[:, :, cur_i : cur_i + valid_chunk_size].clone()
        chunk_mask = all_masks[:, :, cur_i : cur_i + valid_chunk_size]

        actual_overlap = 0
        if global_len > 0:
            # 真实重叠长度 = 已生成的总进度 - 当前段的倒推起点
            # (例如：已生成 81 帧，当前段为了凑 81 帧从第 9 帧开始取，那么重叠帧数就是 81 - 9 = 72 帧！)
            actual_overlap = global_len - cur_i
            
            # 把已经生成的历史画面作为“绝对条件”覆盖到当前段的前面
            # 我们通过 torch.cat 临时拼一下历史结果以便提取
            temp_global_generated = torch.cat(generated_video_chunks, dim=2)
            chunk_cond[:, :, :actual_overlap] = temp_global_generated[:, :, cur_i : global_len]

        print(f"Processing chunk [{cur_i}:{cur_i + valid_chunk_size}] | Overlap context: {actual_overlap} frames...")
        
        # --- 空间分块推理 ---
        chunk_latents = spatial_tiled_process(
            chunk_cond, chunk_mask, tile_num, tile_overlap, prompt_embeds, transformer, vae, noise_scheduler, 
            videoprocessor, vae_scale_factor_spatial, vae_scale_factor_temporal, transformer_patch_size
        )

        # --- 解码当前分段 ---
        with torch.no_grad():
            latents_mean = torch.tensor(vae.config.latents_mean, device=DEVICE, dtype=torch.float32).view(1, vae.config.z_dim, 1, 1, 1)
            latents_std = torch.tensor(vae.config.latents_std, device=DEVICE, dtype=torch.float32).view(1, vae.config.z_dim, 1, 1, 1)
            chunk_latents = chunk_latents.float() * latents_std + latents_mean
            chunk_latents = chunk_latents.to(vae.dtype)
            video_chunk_tensor = vae.decode(chunk_latents, return_dict=False)[0]

            video_chunk_tensor = (video_chunk_tensor / 2 + 0.5).clamp(0, 1)

        # 保存并剔除重复片段
        if global_len == 0:
            generated_video_chunks.append(video_chunk_tensor)
            global_len += video_chunk_tensor.shape[2]
        else:
            # 严格剔除历史重叠帧
            new_frames = video_chunk_tensor[:, :, actual_overlap:]
            generated_video_chunks.append(new_frames)
            global_len += new_frames.shape[2]

        # 保护机制：如果因为取整或视频极短导致无法前进，跳出避免死循环
        if global_len > 0 and actual_overlap >= valid_chunk_size:
            print("Warning: Chunk progression stuck due to temporal scaling limits. Exiting loop.")
            break

    # 拼接所有时间分段
    final_video = torch.cat(generated_video_chunks, dim=2)

    if pad_h > 0 or pad_w > 0:
        print(f"Removing padding, restoring resolution to {w_orig}x{h_orig}...")
        final_video = final_video[:, :, :, :h_orig, :w_orig]
    
    print("\nExporting final video...")
    
    # 1. 取出 batch 0，形状变为 [C, F, H, W]
    video_tensor = final_video[0]
    
    # 2. 转换为 numpy 并调整维度顺序为 [F, H, W, C]
    video_np = video_tensor.permute(1, 2, 3, 0).cpu().float().numpy()
    video_left_np = frames_left[0].permute(1, 2, 3, 0).cpu().float().numpy()


    frames_sbs = np.concatenate([video_left_np, video_np], axis=2)
    frames_sbs_path = os.path.join(save_dir, f"{video_name}_sbs.mp4")
    frames_sbs_frames_list = [frames_sbs[i] for i in range(frames_sbs.shape[0])]
    # print(frames_sbs_frames_list[0].shape)
    export_to_video(frames_sbs_frames_list, frames_sbs_path, fps=int(fps))


    video_left_np[:, :, :, 1] = 0
    video_left_np[:, :, :, 2] = 0
    video_np[:, :, :, 0] = 0

    vid_anaglyph = video_left_np + video_np
    vid_anaglyph_path = os.path.join(save_dir, f"{video_name}_anaglyph.mp4")
    vid_anaglyph_frames_list = [vid_anaglyph[i] for i in range(vid_anaglyph.shape[0])]

    export_to_video(vid_anaglyph_frames_list, vid_anaglyph_path, fps=int(fps))


if __name__ == "__main__":
    Fire(main)
