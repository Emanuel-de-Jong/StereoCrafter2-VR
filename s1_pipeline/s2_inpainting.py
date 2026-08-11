import os
import sys
import math
from pathlib import Path
import torch
import torch.nn.functional as F
import numpy as np

sys.path.append(str(Path(__file__).resolve().parent.parent))

import s0_utils.global_params as g
from s0_utils.helpers import RawVideoWriter, cleanup_cuda, should_skip_output
from s0_utils.monitor import monitor_step

from PIL import Image
from decord import VideoReader, cpu
from diffusers import WanVACETransformer3DModel, AutoencoderKLWan
from diffusers.video_processor import VideoProcessor
from transformers import AutoTokenizer, UMT5EncoderModel
import ftfy
import html
import re
import random
from fire import Fire

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DTYPE = torch.bfloat16
PROMPT = ""
FP8_STATE_FILE = "diffusion_pytorch_model_fp8.pt"


def main(
    pre_trained_path: str = str(g.WAN_WEIGHTS_PATH),
    transformer_path: str = str(g.STEREOCRAFTER_WEIGHTS_PATH),
    input_video_path: str = str(g.OUTPUTS_DIR / "vid_1_splatting.mkv"),
    source_video_path: str = str(g.INPUTS_DIR / "vid.mp4"),
    output_path: str = str(g.OUTPUTS_DIR),
    output_video_path: str | None = None,
    anaglyph_video_path: str | None = None,
    frames_chunk: int = 25,
    frames_overlap: int = 8,
    tile_overlap: int = 128,
    tile_num: int = 3,
    inference_steps: int = 10,
    inpaint_scale: float = 1.0,
    transformer_dtype: str = "fp8",
    transformer_cpu_offload: str = "none",
    vae_cpu_offload: str = "manual",
    scene_cut_threshold: float = 0.22,
    output_crf: int = g.ENCODE_CRF,
    output_preset: str = g.ENCODE_PRESET,
    seed: int = 0,
    overwrite: bool = False,
):
    frames_sbs_path, vid_anaglyph_path = get_inpainting_output_paths(
        input_video_path, output_path, output_video_path, anaglyph_video_path
    )

    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    os.makedirs(os.path.dirname(frames_sbs_path) or ".", exist_ok=True)
    os.makedirs(os.path.dirname(vid_anaglyph_path) or ".", exist_ok=True)

    if should_skip_output(frames_sbs_path, overwrite):
        return

    tokenizer = AutoTokenizer.from_pretrained(pre_trained_path, subfolder="tokenizer")
    text_encoder = UMT5EncoderModel.from_pretrained(
        pre_trained_path, subfolder="text_encoder", torch_dtype=DTYPE
    ).to(DEVICE)
    text_encoder.eval()
    text_encoder.requires_grad_(False)

    print("Encoding prompt...")
    with torch.inference_mode():
        prompt_embeds, _ = encode_prompt(
            [PROMPT],
            do_classifier_free_guidance=False,
            max_sequence_length=226,
            device=DEVICE,
            dtype=DTYPE,
            tokenizer=tokenizer,
            text_encoder=text_encoder,
        )

    text_encoder.to("cpu")
    del text_encoder, tokenizer
    cleanup_cuda()

    vae = AutoencoderKLWan.from_pretrained(
        pre_trained_path, subfolder="vae", torch_dtype=DTYPE, low_cpu_mem_usage=True
    ).to(DEVICE)
    enable_vae_memory_features(vae)

    transformer = load_transformer(
        transformer_path, transformer_dtype, transformer_cpu_offload
    )

    transformer.eval()
    vae.eval()

    vae.requires_grad_(False)
    transformer.requires_grad_(False)

    if isinstance(vae_cpu_offload, str):
        vae_cpu_offload = vae_cpu_offload.lower()
    if vae_cpu_offload not in [None, "none", "cuda", "false", "manual"]:
        raise ValueError(f"Unknown vae cpu offload option: {vae_cpu_offload}")
    if vae_cpu_offload in [None, "none", "cuda", "false"]:
        vae_cpu_offload = "none"

    videoprocessor = VideoProcessor(vae_scale_factor=vae.config.scale_factor_spatial)
    transformer_patch_size = transformer.config.patch_size[1]
    vae_scale_factor_temporal = 2 ** sum(vae.temperal_downsample)
    vae_scale_factor_spatial = 2 ** len(vae.temperal_downsample)

    print("Loading video...")
    video_reader = VideoReader(input_video_path, ctx=cpu(0))
    source_video_reader = VideoReader(source_video_path, ctx=cpu(0))
    fps = video_reader.get_avg_fps()
    source_fps = source_video_reader.get_avg_fps()
    total_frames = min(len(video_reader), len(source_video_reader))

    if abs(fps - source_fps) > 0.01:
        raise ValueError(
            f"Condition and source FPS do not match: {fps:.3f} != {source_fps:.3f}"
        )

    base = vae_scale_factor_spatial * transformer_patch_size

    if inpaint_scale <= 0:
        raise ValueError(f"inpaint_scale must be greater than 0, got: {inpaint_scale}")

    noise_scheduler = FlowMatchScheduler()
    noise_scheduler.set_timesteps(
        num_inference_steps=inference_steps, denoising_strength=1.0
    )

    scene_ranges = detect_scene_ranges(
        source_video_reader, total_frames, scene_cut_threshold
    )
    print(
        f"Starting Temporal Chunking inference (Total Frames: {total_frames}, Scenes: {len(scene_ranges)})..."
    )

    global_len = 0
    scene_index = 0
    generated_context = None
    sbs_writer = None
    anaglyph_writer = None

    while global_len < total_frames:
        scene_start, scene_end = scene_ranges[scene_index]
        at_scene_start = global_len == scene_start

        if at_scene_start:
            cur_i = scene_start
            cur_chunk_size = min(frames_chunk, scene_end - scene_start)
            valid_chunk_size = (
                math.ceil((cur_chunk_size - 1) / vae_scale_factor_temporal)
                * vae_scale_factor_temporal
                + 1
            )
            generated_context = None
        else:
            cur_i = global_len - frames_overlap
            cur_chunk_size = min(frames_chunk, scene_end - cur_i)
            valid_chunk_size = (
                math.ceil((cur_chunk_size - 1) / vae_scale_factor_temporal)
                * vae_scale_factor_temporal
                + 1
            )

        frames_left, chunk_mask, chunk_cond = load_video_chunk(
            video_reader,
            source_video_reader,
            cur_i,
            cur_i + valid_chunk_size,
            scene_end - 1,
            inpaint_scale,
            base,
        )

        h_orig, w_orig = chunk_cond.shape[3], chunk_cond.shape[4]
        pad_h, pad_w = get_tiling_padding(h_orig, w_orig, tile_num, tile_overlap, base)

        if pad_h > 0 or pad_w > 0:
            print(
                f"Padding chunk resolution from {w_orig}x{h_orig} to {w_orig + pad_w}x{h_orig + pad_h} to perfectly match Tiling output."
            )
            chunk_cond, chunk_mask = pad_video_chunk(
                chunk_cond, chunk_mask, pad_h, pad_w
            )

        actual_overlap = 0
        if not at_scene_start:
            actual_overlap = global_len - cur_i
            if generated_context is not None:
                context_size = min(actual_overlap, generated_context.shape[2])
                chunk_cond[
                    :,
                    :,
                    actual_overlap - context_size : actual_overlap,
                    :h_orig,
                    :w_orig,
                ] = generated_context[:, :, -context_size:].to(chunk_cond.device)
                chunk_mask[
                    :,
                    :,
                    actual_overlap - context_size : actual_overlap,
                    :h_orig,
                    :w_orig,
                ] = 0

        print(
            f"Processing chunk [{cur_i}:{cur_i + valid_chunk_size}] | Overlap context: {actual_overlap} frames..."
        )

        chunk_latents = spatial_tiled_process(
            chunk_cond,
            chunk_mask,
            tile_num,
            tile_overlap,
            prompt_embeds,
            transformer,
            vae,
            noise_scheduler,
            videoprocessor,
            vae_scale_factor_spatial,
            vae_scale_factor_temporal,
            transformer_patch_size,
            vae_cpu_offload,
        )

        if vae_cpu_offload == "manual":
            vae.to(DEVICE)

        with torch.inference_mode():
            latents_mean = torch.tensor(
                vae.config.latents_mean, device=DEVICE, dtype=torch.float32
            ).view(1, vae.config.z_dim, 1, 1, 1)
            latents_std = torch.tensor(
                vae.config.latents_std, device=DEVICE, dtype=torch.float32
            ).view(1, vae.config.z_dim, 1, 1, 1)
            chunk_latents = chunk_latents.float() * latents_std + latents_mean
            chunk_latents = chunk_latents.to(vae.dtype)
            video_chunk_tensor = vae.decode(chunk_latents, return_dict=False)[0]

            video_chunk_tensor = (video_chunk_tensor / 2 + 0.5).clamp(0, 1)

        if vae_cpu_offload == "manual":
            vae.to("cpu")
            cleanup_cuda()

        video_chunk_tensor = video_chunk_tensor.cpu()
        output_frame_count = video_chunk_tensor.shape[2]
        preserve_mask = (chunk_mask[:, :, :output_frame_count].float() >= 0.5).to(
            video_chunk_tensor.dtype
        )
        video_chunk_tensor = video_chunk_tensor * preserve_mask + chunk_cond[
            :, :, :output_frame_count
        ].float() * (1.0 - preserve_mask)

        del (
            chunk_latents,
            latents_mean,
            latents_std,
            chunk_cond,
            chunk_mask,
            preserve_mask,
        )

        if pad_h > 0 or pad_w > 0:
            video_chunk_tensor = video_chunk_tensor[:, :, :, :h_orig, :w_orig]

        video_chunk_tensor = video_chunk_tensor[:, :, :cur_chunk_size]
        frames_left = frames_left[:, :, :cur_chunk_size]

        generated_context = video_chunk_tensor
        new_frames = video_chunk_tensor[:, :, actual_overlap:]
        new_left_frames = frames_left[:, :, actual_overlap:].cpu()

        if sbs_writer is None:
            output_height = new_frames.shape[3]
            output_width = new_frames.shape[4]
            sbs_writer = RawVideoWriter(
                frames_sbs_path,
                output_width * 2,
                output_height,
                fps,
                codec="ffv1" if frames_sbs_path.lower().endswith(".mkv") else "libx264",
                crf=output_crf,
                preset=output_preset,
            )
            anaglyph_writer = RawVideoWriter(
                vid_anaglyph_path,
                output_width,
                output_height,
                fps,
                crf=output_crf,
                preset=output_preset,
            )

        write_stereo_frames(
            new_left_frames,
            new_frames,
            sbs_writer,
            anaglyph_writer,
        )
        global_len += new_frames.shape[2]

        del video_chunk_tensor, frames_left, new_frames, new_left_frames

        cleanup_cuda()

        if global_len >= scene_end:
            global_len = scene_end
            scene_index += 1

    if sbs_writer is not None:
        sbs_writer.close()
    if anaglyph_writer is not None:
        anaglyph_writer.close()


def write_stereo_frames(left_frames, right_frames, sbs_writer, anaglyph_writer):
    left_frames = left_frames[0].permute(1, 2, 3, 0).float().numpy()
    right_frames = right_frames[0].permute(1, 2, 3, 0).float().numpy()

    for frame_index in range(len(right_frames)):
        left_frame = np.clip(left_frames[frame_index], 0.0, 1.0)
        right_frame = np.clip(right_frames[frame_index], 0.0, 1.0)
        sbs_writer.write(np.concatenate([left_frame, right_frame], axis=1))

        anaglyph_frame = np.zeros_like(left_frame)
        anaglyph_frame[:, :, 0] = left_frame[:, :, 0]
        anaglyph_frame[:, :, 1:] = right_frame[:, :, 1:]
        anaglyph_writer.write(anaglyph_frame)


def detect_scene_ranges(video_reader, total_frames, threshold):
    if total_frames <= 1 or threshold <= 0:
        return [(0, total_frames)]

    scene_starts = [0]
    previous_frame = video_reader.get_batch([0]).asnumpy()[0]
    previous_frame = F.interpolate(
        torch.from_numpy(previous_frame).permute(2, 0, 1).unsqueeze(0).float(),
        size=(64, 64),
        mode="area",
    )[0].mean(0)

    for frame_index in range(1, total_frames):
        current_frame = video_reader.get_batch([frame_index]).asnumpy()[0]
        current_frame = F.interpolate(
            torch.from_numpy(current_frame).permute(2, 0, 1).unsqueeze(0).float(),
            size=(64, 64),
            mode="area",
        )[0].mean(0)
        frame_difference = (
            float(torch.mean(torch.abs(current_frame - previous_frame))) / 255.0
        )
        if frame_difference >= threshold:
            scene_starts.append(frame_index)
        previous_frame = current_frame

    scene_starts.append(total_frames)
    return [
        (scene_starts[index], scene_starts[index + 1])
        for index in range(len(scene_starts) - 1)
    ]


def get_torch_dtype(dtype):
    if dtype is None:
        return None
    dtype = dtype.lower()
    if dtype in ["auto", "none", "fp8"]:
        return None
    if dtype in ["bfloat16", "bf16"]:
        return torch.bfloat16
    if dtype in ["float16", "fp16"]:
        return torch.float16
    if dtype in ["float32", "fp32"]:
        return torch.float32
    raise ValueError(f"Unknown torch dtype: {dtype}")


def enable_vae_memory_features(vae):
    if hasattr(vae, "enable_slicing"):
        vae.enable_slicing()
    if hasattr(vae, "enable_tiling"):
        vae.enable_tiling()


def load_fp8_transformer(transformer_path):
    import torchao
    from accelerate import init_empty_weights

    config = WanVACETransformer3DModel.load_config(transformer_path)
    with init_empty_weights():
        transformer = WanVACETransformer3DModel.from_config(config)
    state_dict = torch.load(
        os.path.join(transformer_path, FP8_STATE_FILE),
        map_location="cpu",
        weights_only=False,
    )
    transformer.load_state_dict(state_dict, assign=True)
    return transformer


def load_transformer(
    transformer_path, transformer_dtype="auto", transformer_cpu_offload="none"
):
    transformer_dtype_name = (
        transformer_dtype.lower()
        if isinstance(transformer_dtype, str)
        else transformer_dtype
    )
    transformer_dtype = get_torch_dtype(transformer_dtype)
    load_kwargs = {"low_cpu_mem_usage": True}
    if transformer_dtype is not None:
        load_kwargs["torch_dtype"] = transformer_dtype

    fp8_state_path = os.path.join(transformer_path, FP8_STATE_FILE)

    if transformer_dtype_name == "fp8" and not os.path.isfile(fp8_state_path):
        raise FileNotFoundError(
            f"FP8 transformer state file not found: {fp8_state_path}"
        )

    if transformer_dtype_name in ["auto", "fp8"] and os.path.isfile(fp8_state_path):
        transformer = load_fp8_transformer(transformer_path)
    else:
        transformer = WanVACETransformer3DModel.from_pretrained(
            transformer_path, **load_kwargs
        )

    if isinstance(transformer_cpu_offload, str):
        transformer_cpu_offload = transformer_cpu_offload.lower()

    if transformer_cpu_offload in [None, "none", "cuda", "false"]:
        transformer = transformer.to(DEVICE)
    elif transformer_cpu_offload == "group":
        if not hasattr(transformer, "enable_group_offload"):
            raise ValueError(
                "Transformer group offload requires a Diffusers version with enable_group_offload support"
            )
        transformer.enable_group_offload(
            onload_device=DEVICE,
            offload_device=torch.device("cpu"),
            offload_type="block_level",
            num_blocks_per_group=1,
        )
    else:
        raise ValueError(
            f"Unknown transformer cpu offload option: {transformer_cpu_offload}"
        )

    return transformer


class FlowMatchScheduler:
    def __init__(
        self,
    ):
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


def encode_vae_mode(vae, x):
    dist = vae.encode(x).latent_dist
    return dist.mode() if hasattr(dist, "mode") else dist.mean


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
    prompt=None,
    num_videos_per_prompt=1,
    max_sequence_length=226,
    device=None,
    dtype=None,
    tokenizer=None,
    text_encoder=None,
):
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

    prompt_embeds = text_encoder(
        text_input_ids.to(device), mask.to(device)
    ).last_hidden_state
    prompt_embeds = prompt_embeds.to(dtype=dtype, device=device)
    prompt_embeds = [u[:v] for u, v in zip(prompt_embeds, seq_lens)]
    prompt_embeds = torch.stack(
        [
            torch.cat([u, u.new_zeros(max_sequence_length - u.size(0), u.size(1))])
            for u in prompt_embeds
        ],
        dim=0,
    )

    _, seq_len, _ = prompt_embeds.shape
    prompt_embeds = prompt_embeds.repeat(1, num_videos_per_prompt, 1)
    prompt_embeds = prompt_embeds.view(batch_size * num_videos_per_prompt, seq_len, -1)

    return prompt_embeds


def encode_prompt(
    prompt,
    negative_prompt=None,
    do_classifier_free_guidance=True,
    num_videos_per_prompt=1,
    prompt_embeds=None,
    negative_prompt_embeds=None,
    max_sequence_length=226,
    device=None,
    dtype=None,
    tokenizer=None,
    text_encoder=None,
):
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
        negative_prompt = (
            batch_size * [negative_prompt]
            if isinstance(negative_prompt, str)
            else negative_prompt
        )

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
    reference_images=None,
    transformer_patch_size=None,
    vae_scale_factor_temporal=None,
    vae_scale_factor_spatial=None,
) -> torch.Tensor:

    if reference_images is None:
        reference_images = [[None] for _ in range(mask.shape[0])]
    else:
        if mask.shape[0] != len(reference_images):
            raise ValueError(
                f"Batch size of `mask` {mask.shape[0]} and length of `reference_images` {len(reference_images)} does not match."
            )

    mask_list = []
    for mask_, reference_images_batch in zip(mask, reference_images):
        num_channels, num_frames, height, width = mask_.shape
        new_num_frames = (
            num_frames + vae_scale_factor_temporal - 1
        ) // vae_scale_factor_temporal
        new_height = (
            height
            // (vae_scale_factor_spatial * transformer_patch_size)
            * transformer_patch_size
        )
        new_width = (
            width
            // (vae_scale_factor_spatial * transformer_patch_size)
            * transformer_patch_size
        )
        mask_ = mask_[0, :, :, :]
        mask_ = mask_.view(
            num_frames,
            new_height,
            vae_scale_factor_spatial,
            new_width,
            vae_scale_factor_spatial,
        )
        mask_ = mask_.permute(2, 4, 0, 1, 3).flatten(0, 1)
        mask_ = torch.nn.functional.interpolate(
            mask_.unsqueeze(0),
            size=(new_num_frames, new_height, new_width),
            mode="nearest-exact",
        ).squeeze(0)
        num_ref_images = len(reference_images_batch)
        if num_ref_images > 0:
            mask_padding = torch.zeros_like(mask_[:, :num_ref_images, :, :])
            mask_ = torch.cat([mask_padding, mask_], dim=1)
        mask_list.append(mask_)
    return torch.stack(mask_list)


def preprocess_conditions(
    video=None,
    mask=None,
    reference_images=None,
    batch_size: int = 1,
    height: int = 480,
    width: int = 832,
    num_frames: int = 81,
    dtype=None,
    device=None,
    video_processor=None,
    base=None,
):
    if video is not None:
        video_height, video_width = video_processor.get_default_height_width(video[0])

        if video_height * video_width > height * width:
            scale = min(width / video_width, height / video_height)
            video_height, video_width = int(video_height * scale), int(
                video_width * scale
            )

        if video_height % base != 0 or video_width % base != 0:
            video_height = (video_height // base) * base
            video_width = (video_width // base) * base

        assert video_height * video_width <= height * width

        video = video_processor.preprocess_video(video, video_height, video_width)
        image_size = (
            video_height,
            video_width,
        )
    else:
        video = torch.zeros(
            batch_size, 3, num_frames, height, width, dtype=dtype, device=device
        )
        image_size = (height, width)

    if mask is not None:
        mask = video_processor.preprocess_video(mask, image_size[0], image_size[1])
        mask = torch.clamp((mask + 1) / 2, min=0, max=1)
    else:
        mask = torch.ones_like(video)

    video = video.to(dtype=dtype, device=device)
    mask = mask.to(dtype=dtype, device=device)

    if reference_images is None or isinstance(reference_images, Image.Image):
        reference_images = [[reference_images] for _ in range(video.shape[0])]
    elif isinstance(reference_images, (list, tuple)) and isinstance(
        next(iter(reference_images)), Image.Image
    ):
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

    ref_images_lengths = [
        len(reference_images_batch) for reference_images_batch in reference_images
    ]
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
                image,
                size=(new_height, new_width),
                mode="bilinear",
                align_corners=False,
            ).squeeze(0)
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
    reference_images=None,
    device=None,
    vae=None,
) -> torch.Tensor:
    if reference_images is None:
        reference_images = [[None] for _ in range(video.shape[0])]
    else:
        if video.shape[0] != len(reference_images):
            raise ValueError(
                f"Batch size of `video` {video.shape[0]} and length of `reference_images` {len(reference_images)} does not match."
            )

    vae_dtype = vae.dtype
    video = video.to(dtype=vae_dtype)

    latents_mean = torch.tensor(
        vae.config.latents_mean, device=device, dtype=torch.float32
    ).view(1, vae.config.z_dim, 1, 1, 1)
    latents_std = 1.0 / torch.tensor(
        vae.config.latents_std, device=device, dtype=torch.float32
    ).view(1, vae.config.z_dim, 1, 1, 1)

    if mask is None:
        latents = encode_vae_mode(vae, video)
        latents = ((latents.float() - latents_mean) * latents_std).to(vae_dtype)
    else:
        mask = torch.where(mask > 0.5, 1.0, 0.0).to(dtype=vae_dtype)
        inactive = video * (1 - mask)
        reactive = video * mask
        inactive = encode_vae_mode(vae, inactive)
        reactive = encode_vae_mode(vae, reactive)
        inactive = ((inactive.float() - latents_mean) * latents_std).to(vae_dtype)
        reactive = ((reactive.float() - latents_mean) * latents_std).to(vae_dtype)
        latents = torch.cat([inactive, reactive], dim=1)

    latent_list = []
    for latent, reference_images_batch in zip(latents, reference_images):
        for reference_image in reference_images_batch:
            assert reference_image.ndim == 3
            reference_image = reference_image.to(dtype=vae_dtype)
            reference_image = reference_image[None, :, None, :, :]
            reference_latent = vae.encode(reference_image).latent_dist.sample()
            reference_latent = (
                (reference_latent.float() - latents_mean) * latents_std
            ).to(vae_dtype)
            reference_latent = reference_latent.squeeze(0)
            reference_latent = torch.cat(
                [reference_latent, torch.zeros_like(reference_latent)], dim=0
            )
            latent = torch.cat([reference_latent.squeeze(0), latent], dim=1)
        latent_list.append(latent)

    return torch.stack(latent_list)


def blend_h(a: torch.Tensor, b: torch.Tensor, overlap_size: int) -> torch.Tensor:
    weight_b = (torch.arange(overlap_size).view(1, 1, 1, 1, -1) / overlap_size).to(
        b.device, dtype=b.dtype
    )
    b[:, :, :, :, :overlap_size] = (1 - weight_b) * a[
        :, :, :, :, -overlap_size:
    ] + weight_b * b[:, :, :, :, :overlap_size]
    return b


def blend_v(a: torch.Tensor, b: torch.Tensor, overlap_size: int) -> torch.Tensor:
    weight_b = (torch.arange(overlap_size).view(1, 1, 1, -1, 1) / overlap_size).to(
        b.device, dtype=b.dtype
    )
    b[:, :, :, :overlap_size, :] = (1 - weight_b) * a[
        :, :, :, -overlap_size:, :
    ] + weight_b * b[:, :, :, :overlap_size, :]
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
    transformer_patch_size,
    vae_cpu_offload="none",
    initial_latents=None,
):
    height, width = cond_frames.shape[3], cond_frames.shape[4]
    num_frames = cond_frames.shape[2]

    if vae_cpu_offload == "manual":
        vae.to(DEVICE)

    with torch.inference_mode():
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

        conditioning_latents = prepare_video_latents(
            condition_video, mask, reference_images, DEVICE, vae
        )
        mask_for_transformer = prepare_masks(
            mask,
            reference_images,
            transformer_patch_size,
            vae_scale_factor_temporal,
            vae_scale_factor_spatial,
        ).to(DEVICE, dtype=DTYPE)
        control_hidden_states = torch.cat(
            [conditioning_latents, mask_for_transformer], dim=1
        ).to(DTYPE)

    del (
        condition_video,
        mask,
        reference_images,
        conditioning_latents,
        mask_for_transformer,
    )

    if vae_cpu_offload == "manual":
        vae.to("cpu")
        cleanup_cuda()

    c = transformer.config.in_channels
    f = (num_frames - 1) // vae_scale_factor_temporal + 1
    h = height // vae_scale_factor_spatial
    w = width // vae_scale_factor_spatial

    if initial_latents is None:
        latents = torch.randn(1, c, f, h, w, device=DEVICE, dtype=DTYPE)
    else:
        latents = initial_latents.to(DEVICE, dtype=DTYPE).clone()

    for timestep in noise_scheduler.timesteps:
        timestep_tensor = timestep.unsqueeze(0).to(DEVICE, dtype=DTYPE)
        with torch.no_grad():
            model_pred = transformer(
                hidden_states=latents,
                timestep=timestep_tensor,
                encoder_hidden_states=prompt_embeds,
                control_hidden_states=control_hidden_states,
                return_dict=False,
            )[0]
        latents = noise_scheduler.step(model_pred, timestep, latents)
        del model_pred, timestep_tensor

    del control_hidden_states

    return latents


def spatial_tiled_process(
    cond_frames,
    mask_frames,
    tile_num,
    tile_overlap,
    prompt_embeds,
    transformer,
    vae,
    noise_scheduler,
    videoprocessor,
    vae_scale_factor_spatial,
    vae_scale_factor_temporal,
    transformer_patch_size,
    vae_cpu_offload="none",
):
    if tile_num == 1:
        return run_wan_pipeline(
            cond_frames,
            mask_frames,
            prompt_embeds,
            transformer,
            vae,
            noise_scheduler,
            videoprocessor,
            vae_scale_factor_spatial,
            vae_scale_factor_temporal,
            transformer_patch_size,
            vae_cpu_offload,
        )

    height = cond_frames.shape[3]
    width = cond_frames.shape[4]

    base = vae_scale_factor_spatial * transformer_patch_size
    tile_size = (
        int((height + tile_overlap * (tile_num - 1)) / tile_num) // base * base,
        int((width + tile_overlap * (tile_num - 1)) / tile_num) // base * base,
    )
    tile_stride = (tile_size[0] - tile_overlap, tile_size[1] - tile_overlap)
    latent_frames = (cond_frames.shape[2] - 1) // vae_scale_factor_temporal + 1
    latent_tile_height = tile_size[0] // vae_scale_factor_spatial
    latent_tile_width = tile_size[1] // vae_scale_factor_spatial
    latent_stride_height = tile_stride[0] // vae_scale_factor_spatial
    latent_stride_width = tile_stride[1] // vae_scale_factor_spatial

    cols = []
    noise_rows = []
    for row_index in range(tile_num):
        rows = []
        current_noise_row = []
        left_noise = None
        for column_index in range(tile_num):
            h_start = min(row_index * tile_stride[0], height - tile_size[0])
            w_start = min(column_index * tile_stride[1], width - tile_size[1])

            cond_tile = cond_frames[
                :,
                :,
                :,
                h_start : h_start + tile_size[0],
                w_start : w_start + tile_size[1],
            ]
            mask_tile = mask_frames[
                :,
                :,
                :,
                h_start : h_start + tile_size[0],
                w_start : w_start + tile_size[1],
            ]
            initial_latents = torch.randn(
                1,
                transformer.config.in_channels,
                latent_frames,
                latent_tile_height,
                latent_tile_width,
                device=DEVICE,
                dtype=DTYPE,
            )
            if row_index > 0:
                initial_latents[
                    :, :, :, : latent_tile_height - latent_stride_height
                ] = noise_rows[row_index - 1][column_index].to(DEVICE)
            if left_noise is not None:
                initial_latents[
                    :, :, :, :, : latent_tile_width - latent_stride_width
                ] = left_noise.to(DEVICE)

            right_noise = (
                initial_latents[:, :, :, :, latent_stride_width:].cpu().clone()
            )
            bottom_noise = initial_latents[:, :, :, latent_stride_height:].cpu().clone()

            tile_latent = run_wan_pipeline(
                cond_tile,
                mask_tile,
                prompt_embeds,
                transformer,
                vae,
                noise_scheduler,
                videoprocessor,
                vae_scale_factor_spatial,
                vae_scale_factor_temporal,
                transformer_patch_size,
                vae_cpu_offload,
                initial_latents,
            )
            rows.append(tile_latent)
            current_noise_row.append(bottom_noise)
            left_noise = right_noise
            del cond_tile, mask_tile, initial_latents, right_noise, bottom_noise
            cleanup_cuda()
        cols.append(rows)
        noise_rows.append(current_noise_row)

    latent_stride = (
        tile_stride[0] // vae_scale_factor_spatial,
        tile_stride[1] // vae_scale_factor_spatial,
    )
    latent_overlap = (
        tile_overlap // vae_scale_factor_spatial,
        tile_overlap // vae_scale_factor_spatial,
    )

    results_cols = []
    for row_index, rows in enumerate(cols):
        results_rows = []
        for column_index, tile in enumerate(rows):
            if row_index > 0:
                tile = blend_v(
                    cols[row_index - 1][column_index], tile, latent_overlap[0]
                )
            if column_index > 0:
                tile = blend_h(rows[column_index - 1], tile, latent_overlap[1])
            results_rows.append(tile)
        results_cols.append(results_rows)

    pixels = []
    for row_index, rows in enumerate(results_cols):
        for column_index, tile in enumerate(rows):
            if row_index < len(results_cols) - 1:
                tile = tile[:, :, :, : latent_stride[0], :]
            if column_index < len(rows) - 1:
                tile = tile[:, :, :, :, : latent_stride[1]]
            rows[column_index] = tile
        pixels.append(torch.cat(rows, dim=4))

    return torch.cat(pixels, dim=3)


def resize_video_tensor(video_tensor, height, width, mode="bilinear"):
    batch_size, channels, num_frames = video_tensor.shape[:3]
    video_tensor = video_tensor.permute(0, 2, 1, 3, 4).reshape(
        batch_size * num_frames, channels, video_tensor.shape[3], video_tensor.shape[4]
    )

    if mode in ["linear", "bilinear", "bicubic", "trilinear"]:
        video_tensor = F.interpolate(
            video_tensor, size=(height, width), mode=mode, align_corners=False
        )
    else:
        video_tensor = F.interpolate(video_tensor, size=(height, width), mode=mode)

    return video_tensor.reshape(
        batch_size, num_frames, channels, height, width
    ).permute(0, 2, 1, 3, 4)


def load_video_chunk(
    video_reader,
    source_video_reader,
    start_frame,
    end_frame,
    max_frame_index,
    inpaint_scale,
    base,
):
    frame_indices = [
        min(frame_index, max_frame_index)
        for frame_index in range(start_frame, end_frame)
    ]
    frames = video_reader.get_batch(frame_indices)
    frames = (
        torch.from_numpy(frames.asnumpy()).permute(3, 0, 1, 2).unsqueeze(0).float()
        / 255.0
    )

    height, width = frames.shape[3], frames.shape[4] // 2
    all_masks = frames[:, :, :, :, :width].clone()
    all_frames = frames[:, :, :, :, width:].clone()
    del frames

    source_frames = source_video_reader.get_batch(frame_indices)
    frames_left = (
        torch.from_numpy(source_frames.asnumpy())
        .permute(3, 0, 1, 2)
        .unsqueeze(0)
        .float()
        / 255.0
    )

    if inpaint_scale != 1.0:
        scaled_height = max(base, int(height * inpaint_scale) // base * base)
        scaled_width = max(base, int(width * inpaint_scale) // base * base)
        all_masks = resize_video_tensor(
            all_masks, scaled_height, scaled_width, mode="nearest"
        )
        all_frames = resize_video_tensor(all_frames, scaled_height, scaled_width)

    if frames_left.shape[3:] != all_frames.shape[3:]:
        frames_left = resize_video_tensor(
            frames_left,
            all_frames.shape[3],
            all_frames.shape[4],
            mode="bicubic",
        )

    all_frames = all_frames * (1.0 - all_masks) + 0.5 * all_masks

    return frames_left, all_masks, all_frames


def get_tiling_padding(height, width, tile_num, tile_overlap, base):
    min_tile_h = (height + tile_overlap * (tile_num - 1)) / tile_num
    tile_size_h = math.ceil(min_tile_h / base) * base

    min_tile_w = (width + tile_overlap * (tile_num - 1)) / tile_num
    tile_size_w = math.ceil(min_tile_w / base) * base

    tile_stride_h = tile_size_h - tile_overlap
    tile_stride_w = tile_size_w - tile_overlap

    target_h = tile_stride_h * (tile_num - 1) + tile_size_h
    target_w = tile_stride_w * (tile_num - 1) + tile_size_w

    return target_h - height, target_w - width


def pad_video_chunk(all_frames, all_masks, pad_h, pad_w):
    if pad_h <= 0 and pad_w <= 0:
        return all_frames, all_masks

    frames_4d = all_frames[0].permute(1, 0, 2, 3)
    frames_4d = F.pad(frames_4d, (0, pad_w, 0, pad_h), mode="replicate")
    all_frames = frames_4d.permute(1, 0, 2, 3).unsqueeze(0)
    all_masks = F.pad(all_masks, (0, pad_w, 0, pad_h), mode="constant", value=0)

    return all_frames, all_masks


def get_inpainting_output_paths(
    input_video_path, output_path, output_video_path=None, anaglyph_video_path=None
):
    video_name = os.path.splitext(os.path.basename(input_video_path))[0].replace(
        "_1_splatting", ""
    )
    if output_video_path is None:
        output_video_path = os.path.join(output_path, f"{video_name}_2_sbs.mp4")
    if anaglyph_video_path is None:
        anaglyph_video_path = os.path.join(output_path, f"{video_name}_2_anaglyph.mp4")

    return output_video_path, anaglyph_video_path


if __name__ == "__main__":
    Fire(monitor_step("Step 2 - Inpainting")(main))
