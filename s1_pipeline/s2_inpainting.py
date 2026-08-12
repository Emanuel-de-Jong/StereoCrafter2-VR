import math
import os
import random
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from decord import VideoReader, cpu
from diffusers import AutoencoderKLWan, WanVACETransformer3DModel
from diffusers.video_processor import VideoProcessor
from fire import Fire
from transformers import AutoTokenizer, UMT5EncoderModel

sys.path.append(str(Path(__file__).resolve().parent.parent))

import s0_utils.global_params as g
from s0_utils.helpers import RawVideoWriter, cleanup_cuda, should_skip_output
from s0_utils.monitor import monitor_step

DEVICE = torch.device("cuda")
DTYPE = torch.bfloat16
FP8_STATE_FILE = "diffusion_pytorch_model_fp8.pt"
TILE_NUM = 2


def main(
    pre_trained_path: str = str(g.WAN_WEIGHTS_PATH),
    transformer_path: str = str(g.STEREOCRAFTER_WEIGHTS_PATH),
    input_video_path: str = str(g.OUTPUTS_DIR / "vid_1_splatting.mkv"),
    source_video_path: str = str(g.INPUTS_DIR / "vid.mp4"),
    output_video_path: str = str(g.OUTPUTS_DIR / "vid_2_sbs.mkv"),
    frames_chunk: int = 25,
    frames_overlap: int = 4,
    tile_overlap: int = 128,
    inference_steps: int = 5,
    seed: int = 0,
    overwrite: bool = False,
):
    if should_skip_output(output_video_path, overwrite):
        return
    if frames_chunk <= 0:
        raise ValueError(f"frames_chunk must be greater than 0, got: {frames_chunk}")
    if frames_overlap < 0 or frames_overlap >= frames_chunk:
        raise ValueError(
            f"frames_overlap must be between 0 and frames_chunk - 1, got: {frames_overlap}"
        )
    if tile_overlap <= 0:
        raise ValueError(f"tile_overlap must be greater than 0, got: {tile_overlap}")
    if inference_steps <= 0:
        raise ValueError(
            f"inference_steps must be greater than 0, got: {inference_steps}"
        )

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.makedirs(os.path.dirname(output_video_path) or ".", exist_ok=True)

    prompt_embeds = encode_empty_prompt(pre_trained_path)
    vae = AutoencoderKLWan.from_pretrained(
        pre_trained_path,
        subfolder="vae",
        torch_dtype=DTYPE,
        low_cpu_mem_usage=True,
    )
    if hasattr(vae, "enable_slicing"):
        vae.enable_slicing()
    if hasattr(vae, "enable_tiling"):
        vae.enable_tiling()
    vae.eval()
    vae.requires_grad_(False)

    transformer = load_fp8_transformer(transformer_path).to(DEVICE)
    transformer.eval()
    transformer.requires_grad_(False)

    video_processor = VideoProcessor(vae_scale_factor=vae.config.scale_factor_spatial)
    transformer_patch_size = transformer.config.patch_size[1]
    vae_scale_factor_temporal = 2 ** sum(vae.temperal_downsample)
    vae_scale_factor_spatial = 2 ** len(vae.temperal_downsample)
    base = vae_scale_factor_spatial * transformer_patch_size
    noise_scheduler = FlowMatchScheduler()
    noise_scheduler.set_timesteps(inference_steps)

    print("Loading video...", flush=True)
    video_reader = VideoReader(input_video_path, ctx=cpu(0))
    source_video_reader = VideoReader(source_video_path, ctx=cpu(0))
    fps = video_reader.get_avg_fps()
    source_frame_indices = get_source_frame_indices(
        len(video_reader),
        fps,
        source_video_reader.get_avg_fps(),
        len(source_video_reader),
    )

    total_frames = len(video_reader)
    if total_frames <= 0:
        raise ValueError(f"Input video has no frames: {input_video_path}")
    print(f"Starting inpainting (Total Frames: {total_frames})...", flush=True)
    output_writer = None
    written_frames = 0

    while written_frames < total_frames:
        chunk_start = max(written_frames - frames_overlap, 0)
        actual_overlap = written_frames - chunk_start
        chunk_size = min(frames_chunk, total_frames - chunk_start)
        valid_chunk_size = (
            math.ceil((chunk_size - 1) / vae_scale_factor_temporal)
            * vae_scale_factor_temporal
            + 1
        )
        left_frames, mask_frames, condition_frames = load_video_chunk(
            video_reader,
            source_video_reader,
            source_frame_indices,
            chunk_start,
            chunk_start + valid_chunk_size,
            total_frames - 1,
        )

        original_height = condition_frames.shape[3]
        original_width = condition_frames.shape[4]
        pad_height, pad_width = get_tiling_padding(
            original_height,
            original_width,
            tile_overlap,
            base,
        )
        condition_frames, mask_frames = pad_video_chunk(
            condition_frames,
            mask_frames,
            pad_height,
            pad_width,
        )

        print(
            f"Processing chunk [{chunk_start}:{chunk_start + valid_chunk_size}] | Overlap: {actual_overlap} frames...",
            flush=True,
        )
        chunk_latents = spatial_tiled_process(
            condition_frames,
            mask_frames,
            tile_overlap,
            prompt_embeds,
            transformer,
            vae,
            noise_scheduler,
            video_processor,
            vae_scale_factor_spatial,
            vae_scale_factor_temporal,
            transformer_patch_size,
        )

        vae.to(DEVICE)
        with torch.inference_mode():
            latents_mean = torch.tensor(
                vae.config.latents_mean,
                device=DEVICE,
                dtype=torch.float32,
            ).view(1, vae.config.z_dim, 1, 1, 1)
            latents_std = torch.tensor(
                vae.config.latents_std,
                device=DEVICE,
                dtype=torch.float32,
            ).view(1, vae.config.z_dim, 1, 1, 1)
            chunk_latents = chunk_latents.float() * latents_std + latents_mean
            video_chunk = vae.decode(
                chunk_latents.to(vae.dtype),
                return_dict=False,
            )[0]
            video_chunk = (video_chunk / 2 + 0.5).clamp(0, 1).cpu()

        vae.to("cpu")
        del chunk_latents, latents_mean, latents_std
        cleanup_cuda()

        output_frame_count = video_chunk.shape[2]
        preserve_mask = (mask_frames[:, :, :output_frame_count].cpu() >= 0.5).to(
            video_chunk.dtype
        )
        video_chunk = video_chunk * preserve_mask + condition_frames.cpu()[
            :, :, :output_frame_count
        ] * (1.0 - preserve_mask)
        video_chunk = video_chunk[:, :, :chunk_size, :original_height, :original_width]
        left_frames = left_frames[:, :, :chunk_size]

        new_right_frames = video_chunk[:, :, actual_overlap:]
        new_left_frames = left_frames[:, :, actual_overlap:]
        if output_writer is None:
            output_writer = RawVideoWriter(
                output_video_path,
                original_width * 2,
                original_height,
                fps,
                codec="ffv1",
            )
        write_stereo_frames(new_left_frames, new_right_frames, output_writer)
        written_frames += new_right_frames.shape[2]

        del (
            left_frames,
            mask_frames,
            condition_frames,
            preserve_mask,
            video_chunk,
            new_right_frames,
            new_left_frames,
        )
        cleanup_cuda()

    if output_writer is not None:
        output_writer.close()


def encode_empty_prompt(pre_trained_path):
    tokenizer = AutoTokenizer.from_pretrained(
        pre_trained_path,
        subfolder="tokenizer",
    )
    text_encoder = UMT5EncoderModel.from_pretrained(
        pre_trained_path,
        subfolder="text_encoder",
        torch_dtype=DTYPE,
    ).to(DEVICE)
    text_encoder.eval()
    text_encoder.requires_grad_(False)

    print("Encoding prompt...", flush=True)
    text_inputs = tokenizer(
        [""],
        padding="max_length",
        max_length=226,
        truncation=True,
        add_special_tokens=True,
        return_attention_mask=True,
        return_tensors="pt",
    )
    with torch.inference_mode():
        prompt_embeds = text_encoder(
            text_inputs.input_ids.to(DEVICE),
            text_inputs.attention_mask.to(DEVICE),
        ).last_hidden_state.to(dtype=DTYPE)
        prompt_embeds = prompt_embeds * text_inputs.attention_mask.to(
            DEVICE,
            dtype=DTYPE,
        ).unsqueeze(-1)

    text_encoder.to("cpu")
    del text_encoder, tokenizer, text_inputs
    cleanup_cuda()
    return prompt_embeds


def load_fp8_transformer(transformer_path):
    import torchao
    from accelerate import init_empty_weights

    config = WanVACETransformer3DModel.load_config(transformer_path)
    with init_empty_weights():
        transformer = WanVACETransformer3DModel.from_config(config)
    state_path = os.path.join(transformer_path, FP8_STATE_FILE)
    if not os.path.isfile(state_path):
        raise FileNotFoundError(f"FP8 transformer state file not found: {state_path}")
    state_dict = torch.load(
        state_path,
        map_location="cpu",
        weights_only=False,
    )
    transformer.load_state_dict(state_dict, assign=True)
    return transformer


class FlowMatchScheduler:
    def __init__(self):
        self.sigmas = None
        self.timesteps = None

    def set_timesteps(self, num_inference_steps):
        sigmas = torch.linspace(1.0, 0.0, num_inference_steps + 1)[:-1]
        self.sigmas = 5 * sigmas / (1 + 4 * sigmas)
        self.timesteps = self.sigmas * 1000

    def step(self, model_output, timestep, sample):
        timestep_id = torch.argmin((self.timesteps - timestep.cpu()).abs())
        sigma = self.sigmas[timestep_id]
        if timestep_id + 1 >= len(self.timesteps):
            next_sigma = 0
        else:
            next_sigma = self.sigmas[timestep_id + 1]
        return sample + model_output * (next_sigma - sigma)


def encode_vae_mode(vae, video):
    distribution = vae.encode(video).latent_dist
    return distribution.mode() if hasattr(distribution, "mode") else distribution.mean


def prepare_masks(
    mask,
    transformer_patch_size,
    vae_scale_factor_temporal,
    vae_scale_factor_spatial,
):
    prepared_masks = []
    for batch_mask in mask:
        _num_channels, num_frames, height, width = batch_mask.shape
        latent_frames = (
            num_frames + vae_scale_factor_temporal - 1
        ) // vae_scale_factor_temporal
        latent_height = (
            height
            // (vae_scale_factor_spatial * transformer_patch_size)
            * transformer_patch_size
        )
        latent_width = (
            width
            // (vae_scale_factor_spatial * transformer_patch_size)
            * transformer_patch_size
        )
        batch_mask = batch_mask[0].view(
            num_frames,
            latent_height,
            vae_scale_factor_spatial,
            latent_width,
            vae_scale_factor_spatial,
        )
        batch_mask = batch_mask.permute(2, 4, 0, 1, 3).flatten(0, 1)
        batch_mask = F.interpolate(
            batch_mask.unsqueeze(0),
            size=(latent_frames, latent_height, latent_width),
            mode="nearest-exact",
        ).squeeze(0)
        prepared_masks.append(batch_mask)
    return torch.stack(prepared_masks)


def prepare_video_latents(video, mask, vae):
    vae_dtype = vae.dtype
    video = video.to(dtype=vae_dtype)
    mask = torch.where(mask > 0.5, 1.0, 0.0).to(dtype=vae_dtype)
    latents_mean = torch.tensor(
        vae.config.latents_mean,
        device=DEVICE,
        dtype=torch.float32,
    ).view(1, vae.config.z_dim, 1, 1, 1)
    latents_std = 1.0 / torch.tensor(
        vae.config.latents_std,
        device=DEVICE,
        dtype=torch.float32,
    ).view(1, vae.config.z_dim, 1, 1, 1)
    inactive_latents = encode_vae_mode(vae, video * (1 - mask))
    reactive_latents = encode_vae_mode(vae, video * mask)
    inactive_latents = ((inactive_latents.float() - latents_mean) * latents_std).to(
        vae_dtype
    )
    reactive_latents = ((reactive_latents.float() - latents_mean) * latents_std).to(
        vae_dtype
    )
    return torch.cat([inactive_latents, reactive_latents], dim=1)


def run_wan_pipeline(
    condition_frames,
    mask_frames,
    prompt_embeds,
    transformer,
    vae,
    noise_scheduler,
    video_processor,
    vae_scale_factor_spatial,
    vae_scale_factor_temporal,
    transformer_patch_size,
    initial_latents,
):
    height = condition_frames.shape[3]
    width = condition_frames.shape[4]
    vae.to(DEVICE)

    with torch.inference_mode():
        condition_video = video_processor.preprocess_video(
            condition_frames.permute(0, 2, 1, 3, 4),
            height,
            width,
        ).to(device=DEVICE, dtype=DTYPE)
        mask = video_processor.preprocess_video(
            mask_frames.permute(0, 2, 1, 3, 4),
            height,
            width,
        )
        mask = torch.clamp((mask + 1) / 2, 0, 1).to(
            device=DEVICE,
            dtype=DTYPE,
        )
        conditioning_latents = prepare_video_latents(condition_video, mask, vae)
        mask_for_transformer = prepare_masks(
            mask,
            transformer_patch_size,
            vae_scale_factor_temporal,
            vae_scale_factor_spatial,
        ).to(DEVICE, dtype=DTYPE)
        control_hidden_states = torch.cat(
            [conditioning_latents, mask_for_transformer],
            dim=1,
        ).to(DTYPE)

    vae.to("cpu")
    del condition_video, mask, conditioning_latents, mask_for_transformer
    cleanup_cuda()

    latents = initial_latents.to(DEVICE, dtype=DTYPE).clone()
    del initial_latents
    for timestep in noise_scheduler.timesteps:
        timestep_tensor = timestep.unsqueeze(0).to(DEVICE, dtype=DTYPE)
        with torch.inference_mode():
            model_prediction = transformer(
                hidden_states=latents,
                timestep=timestep_tensor,
                encoder_hidden_states=prompt_embeds,
                control_hidden_states=control_hidden_states,
                return_dict=False,
            )[0]
        latents = noise_scheduler.step(model_prediction, timestep, latents)
        del model_prediction, timestep_tensor

    del control_hidden_states
    return latents


def blend_horizontal(left_tile, right_tile, overlap_size):
    right_weight = (torch.arange(overlap_size).view(1, 1, 1, 1, -1) / overlap_size).to(
        right_tile.device, dtype=right_tile.dtype
    )
    right_tile[:, :, :, :, :overlap_size] = (1 - right_weight) * left_tile[
        :, :, :, :, -overlap_size:
    ] + right_weight * right_tile[:, :, :, :, :overlap_size]
    return right_tile


def blend_vertical(top_tile, bottom_tile, overlap_size):
    bottom_weight = (torch.arange(overlap_size).view(1, 1, 1, -1, 1) / overlap_size).to(
        bottom_tile.device, dtype=bottom_tile.dtype
    )
    bottom_tile[:, :, :, :overlap_size, :] = (1 - bottom_weight) * top_tile[
        :, :, :, -overlap_size:, :
    ] + bottom_weight * bottom_tile[:, :, :, :overlap_size, :]
    return bottom_tile


def spatial_tiled_process(
    condition_frames,
    mask_frames,
    tile_overlap,
    prompt_embeds,
    transformer,
    vae,
    noise_scheduler,
    video_processor,
    vae_scale_factor_spatial,
    vae_scale_factor_temporal,
    transformer_patch_size,
):
    height = condition_frames.shape[3]
    width = condition_frames.shape[4]
    base = vae_scale_factor_spatial * transformer_patch_size
    tile_height = (
        int((height + tile_overlap * (TILE_NUM - 1)) / TILE_NUM) // base * base
    )
    tile_width = int((width + tile_overlap * (TILE_NUM - 1)) / TILE_NUM) // base * base
    tile_stride_height = tile_height - tile_overlap
    tile_stride_width = tile_width - tile_overlap
    latent_frames = (condition_frames.shape[2] - 1) // vae_scale_factor_temporal + 1
    latent_tile_height = tile_height // vae_scale_factor_spatial
    latent_tile_width = tile_width // vae_scale_factor_spatial
    latent_stride_height = tile_stride_height // vae_scale_factor_spatial
    latent_stride_width = tile_stride_width // vae_scale_factor_spatial

    tile_rows = []
    noise_rows = []
    for row_index in range(TILE_NUM):
        current_tiles = []
        current_noise_row = []
        left_noise = None
        for column_index in range(TILE_NUM):
            height_start = min(row_index * tile_stride_height, height - tile_height)
            width_start = min(column_index * tile_stride_width, width - tile_width)
            condition_tile = condition_frames[
                :,
                :,
                :,
                height_start : height_start + tile_height,
                width_start : width_start + tile_width,
            ]
            mask_tile = mask_frames[
                :,
                :,
                :,
                height_start : height_start + tile_height,
                width_start : width_start + tile_width,
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
            tile_latents = run_wan_pipeline(
                condition_tile,
                mask_tile,
                prompt_embeds,
                transformer,
                vae,
                noise_scheduler,
                video_processor,
                vae_scale_factor_spatial,
                vae_scale_factor_temporal,
                transformer_patch_size,
                initial_latents,
            )
            current_tiles.append(tile_latents)
            current_noise_row.append(bottom_noise)
            left_noise = right_noise
            del condition_tile, mask_tile, initial_latents, right_noise, bottom_noise
            cleanup_cuda()
        tile_rows.append(current_tiles)
        noise_rows.append(current_noise_row)

    latent_overlap = tile_overlap // vae_scale_factor_spatial
    result_rows = []
    for row_index, current_tiles in enumerate(tile_rows):
        result_tiles = []
        for column_index, tile_latents in enumerate(current_tiles):
            if row_index > 0:
                tile_latents = blend_vertical(
                    tile_rows[row_index - 1][column_index],
                    tile_latents,
                    latent_overlap,
                )
            if column_index > 0:
                tile_latents = blend_horizontal(
                    current_tiles[column_index - 1],
                    tile_latents,
                    latent_overlap,
                )
            if row_index < TILE_NUM - 1:
                tile_latents = tile_latents[:, :, :, :latent_stride_height, :]
            if column_index < TILE_NUM - 1:
                tile_latents = tile_latents[:, :, :, :, :latent_stride_width]
            result_tiles.append(tile_latents)
        result_rows.append(torch.cat(result_tiles, dim=4))

    return torch.cat(result_rows, dim=3)


def load_video_chunk(
    video_reader,
    source_video_reader,
    source_frame_indices,
    start_frame,
    end_frame,
    max_frame_index,
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
    if frames.shape[4] % 2 != 0:
        raise ValueError(f"Splatting video width must be even, got: {frames.shape[4]}")
    width = frames.shape[4] // 2
    masks = frames[:, :, :, :, :width].clone()
    condition_frames = frames[:, :, :, :, width:].clone()
    del frames

    source_frames = source_video_reader.get_batch(
        [source_frame_indices[frame_index] for frame_index in frame_indices]
    )
    left_frames = (
        torch.from_numpy(source_frames.asnumpy())
        .permute(3, 0, 1, 2)
        .unsqueeze(0)
        .float()
        / 255.0
    )
    condition_frames = condition_frames * (1.0 - masks) + 0.5 * masks
    return left_frames, masks, condition_frames


def get_source_frame_indices(
    total_frames,
    condition_fps,
    source_fps,
    source_frame_count,
):
    if condition_fps <= 0 or source_fps <= 0:
        raise ValueError(
            f"Condition and source FPS must be positive: {condition_fps:.3f}, {source_fps:.3f}"
        )
    if source_frame_count <= 0:
        raise ValueError("Source video has no frames")
    return [
        min(round(frame_index * source_fps / condition_fps), source_frame_count - 1)
        for frame_index in range(total_frames)
    ]


def get_tiling_padding(height, width, tile_overlap, base):
    tile_height = (
        math.ceil(((height + tile_overlap * (TILE_NUM - 1)) / TILE_NUM) / base) * base
    )
    tile_width = (
        math.ceil(((width + tile_overlap * (TILE_NUM - 1)) / TILE_NUM) / base) * base
    )
    target_height = (tile_height - tile_overlap) * (TILE_NUM - 1) + tile_height
    target_width = (tile_width - tile_overlap) * (TILE_NUM - 1) + tile_width
    return target_height - height, target_width - width


def pad_video_chunk(frames, masks, pad_height, pad_width):
    if pad_height <= 0 and pad_width <= 0:
        return frames, masks
    frames_4d = frames[0].permute(1, 0, 2, 3)
    frames_4d = F.pad(
        frames_4d,
        (0, pad_width, 0, pad_height),
        mode="replicate",
    )
    frames = frames_4d.permute(1, 0, 2, 3).unsqueeze(0)
    masks = F.pad(
        masks,
        (0, pad_width, 0, pad_height),
        mode="constant",
        value=0,
    )
    return frames, masks


def write_stereo_frames(left_frames, right_frames, output_writer):
    left_frames = left_frames[0].permute(1, 2, 3, 0).numpy()
    right_frames = right_frames[0].permute(1, 2, 3, 0).numpy()
    for frame_index in range(len(right_frames)):
        output_writer.write(
            np.concatenate(
                [left_frames[frame_index], right_frames[frame_index]],
                axis=1,
            )
        )


if __name__ == "__main__":
    Fire(monitor_step("Step 2 - Inpainting")(main))
