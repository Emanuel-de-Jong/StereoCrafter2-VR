import gc
import os
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from decord import VideoReader, cpu
from diffusers.training_utils import set_seed
from fire import Fire

sys.path.append(str(Path(__file__).resolve().parent.parent))

import s0_utils.global_params as g
from s0_utils.helpers import RawVideoWriter, cleanup_cuda, should_skip_output
from s0_utils.monitor import monitor_step
from s0_utils.stereo import ForwardWarpStereo, get_disparity

from dependencies.DepthCrafter.depthcrafter.depth_crafter_ppl import (
    DepthCrafterPipeline,
)
from dependencies.DepthCrafter.depthcrafter.unet import (
    DiffusersUNetSpatioTemporalConditionModelDepthCrafter,
)


def main(
    input_video_path: str = str(g.INPUTS_DIR / "vid.mp4"),
    output_video_path: str = str(g.OUTPUTS_DIR / "vid_1_splatting.mkv"),
    unet_path: str = str(g.DEPTHCRAFTER_WEIGHTS_PATH),
    pre_trained_path: str = str(g.SVD_WEIGHTS_PATH),
    max_disp: float = 20,
    max_disp_reference_width: int = 1920,
    process_length: int = -1,
    batch_size: int = 10,
    num_denoising_steps: int = 5,
    guidance_scale: float = 1.0,
    window_size: int = 56,
    overlap: int = 16,
    max_res: int = 1024,
    target_fps: int = 30,
    seed: int = 42,
    decode_chunk_size: int = 8,
    overwrite: bool = False,
):
    if should_skip_output(output_video_path, overwrite):
        return
    if batch_size <= 0:
        raise ValueError(f"batch_size must be greater than 0, got: {batch_size}")
    if decode_chunk_size <= 0:
        raise ValueError(
            f"decode_chunk_size must be greater than 0, got: {decode_chunk_size}"
        )
    if target_fps == 0 or target_fps < -1:
        raise ValueError(f"target_fps must be -1 or greater than 0, got: {target_fps}")
    os.makedirs(os.path.dirname(output_video_path) or ".", exist_ok=True)

    depthcrafter_demo = DepthCrafterDemo(
        unet_path=unet_path,
        pre_trained_path=pre_trained_path,
    )
    video_depth, depth_fps = depthcrafter_demo.infer(
        input_video_path,
        process_length,
        num_denoising_steps,
        guidance_scale,
        window_size,
        overlap,
        max_res,
        target_fps,
        seed,
        decode_chunk_size,
    )

    print("==> unloading DepthCrafter before splatting", flush=True)
    del depthcrafter_demo
    cleanup_cuda()

    save_depth(
        output_video_path,
        video_depth,
        depth_fps,
        max_disp,
        max_disp_reference_width,
    )
    depth_splatting(
        input_video_path,
        output_video_path,
        video_depth,
        max_disp,
        max_disp_reference_width,
        process_length,
        batch_size,
        target_fps,
    )


def read_video_frames(video_path, process_length, target_fps, max_res):
    print("==> processing video: ", video_path, flush=True)
    video_reader = VideoReader(video_path, ctx=cpu(0))
    first_frame = video_reader.get_batch([0])
    original_height, original_width = first_frame.shape[1:3]
    print(
        "==> original video shape: ",
        (len(video_reader), *first_frame.shape[1:]),
        flush=True,
    )

    height = round(original_height / 64) * 64
    width = round(original_width / 64) * 64
    if max(height, width) > max_res:
        scale = max_res / max(original_height, original_width)
        height = round(original_height * scale / 64) * 64
        width = round(original_width * scale / 64) * 64

    video_reader = VideoReader(
        video_path,
        ctx=cpu(0),
        width=width,
        height=height,
    )
    average_fps = video_reader.get_avg_fps()
    maximum_fps = average_fps if target_fps == -1 else min(target_fps, average_fps)
    stride = max(round(average_fps / maximum_fps), 1)
    fps = average_fps / stride
    frame_indices = list(range(0, len(video_reader), stride))
    if process_length != -1 and process_length < len(frame_indices):
        frame_indices = frame_indices[:process_length]

    print(
        f"==> final processing shape: {(len(frame_indices), height, width, first_frame.shape[3])}, with stride: {stride}",
        flush=True,
    )
    frames = video_reader.get_batch(frame_indices).asnumpy().astype("float32") / 255.0
    return frames, fps, original_height, original_width


class DepthCrafterDemo:
    def __init__(
        self,
        unet_path: str,
        pre_trained_path: str,
    ):
        unet = DiffusersUNetSpatioTemporalConditionModelDepthCrafter.from_pretrained(
            unet_path,
            low_cpu_mem_usage=True,
            torch_dtype=torch.float16,
        )
        self.pipe = DepthCrafterPipeline.from_pretrained(
            pre_trained_path,
            unet=unet,
            torch_dtype=torch.float16,
            variant="fp16",
        )
        self.pipe.enable_model_cpu_offload()
        try:
            self.pipe.enable_xformers_memory_efficient_attention()
        except Exception as error:
            print(error, flush=True)
            print("Xformers is not enabled", flush=True)
        self.pipe.enable_attention_slicing()

    def infer(
        self,
        input_video_path,
        process_length,
        num_denoising_steps,
        guidance_scale,
        window_size,
        overlap,
        max_res,
        target_fps,
        seed,
        decode_chunk_size,
    ):
        set_seed(seed)
        frames, fps, original_height, original_width = read_video_frames(
            input_video_path,
            process_length,
            target_fps,
            max_res,
        )

        print("==> running DepthCrafter depth inference", flush=True)
        with torch.inference_mode():
            depth = self.pipe(
                frames,
                height=frames.shape[1],
                width=frames.shape[2],
                output_type="np",
                guidance_scale=guidance_scale,
                num_inference_steps=num_denoising_steps,
                window_size=min(window_size, len(frames)),
                overlap=min(overlap, max(len(frames) - 1, 0)),
                decode_chunk_size=decode_chunk_size,
            ).frames[0]
        depth = depth.sum(-1) / depth.shape[-1]

        resized_depth = []
        for i in range(0, len(depth), decode_chunk_size):
            depth_tensor = (
                torch.from_numpy(depth[i : i + decode_chunk_size])
                .unsqueeze(1)
                .float()
                .cuda()
            )
            depth_tensor = F.interpolate(
                depth_tensor,
                size=(original_height, original_width),
                mode="bilinear",
                align_corners=False,
            )
            resized_depth.append(depth_tensor.cpu().numpy()[:, 0])
            del depth_tensor
        depth = np.concatenate(resized_depth, axis=0)

        depth_min = float(depth.min())
        depth_range = max(float(depth.max()) - depth_min, 1e-6)
        depth = ((depth - depth_min) / depth_range).astype(np.float32)
        return depth, fps


def save_depth(
    output_video_path,
    video_depth,
    depth_fps,
    max_disp,
    max_disp_reference_width,
):
    save_path = os.path.splitext(output_video_path)[0]
    np.savez_compressed(
        save_path + ".npz",
        depth=video_depth,
        fps=np.float32(depth_fps),
        max_disp=np.float32(max_disp),
        max_disp_reference_width=np.int32(max_disp_reference_width),
    )


def depth_splatting(
    input_video_path,
    output_video_path,
    video_depth,
    max_disp,
    max_disp_reference_width,
    process_length,
    batch_size,
    target_fps,
):
    print("==> loading frames for splatting", flush=True)
    video_reader = VideoReader(input_video_path, ctx=cpu(0))
    original_fps = video_reader.get_avg_fps()
    maximum_fps = original_fps if target_fps == -1 else min(target_fps, original_fps)
    stride = max(round(original_fps / maximum_fps), 1)
    fps = original_fps / stride
    frame_indices = list(range(0, len(video_reader), stride))
    if process_length != -1 and process_length < len(frame_indices):
        frame_indices = frame_indices[:process_length]

    frame_count = min(len(frame_indices), len(video_depth))
    frame_indices = frame_indices[:frame_count]
    video_depth = video_depth[:frame_count]
    if not frame_indices:
        raise ValueError("No frames are available for depth splatting")
    stereo_projector = ForwardWarpStereo().cuda()
    height, width = video_reader.get_batch([frame_indices[0]]).shape[1:3]
    effective_max_disp = max_disp * width / max_disp_reference_width
    print(f"==> effective maximum disparity: {effective_max_disp:.2f}px", flush=True)
    output_writer = RawVideoWriter(
        output_video_path,
        width * 2,
        height,
        fps,
        codec="ffv1",
    )

    for i in range(0, len(frame_indices), batch_size):
        print(
            f"==> splatting frames {i + 1}-{min(i + batch_size, len(frame_indices))} / {len(frame_indices)}",
            flush=True,
        )
        batch_indices = frame_indices[i : i + batch_size]
        batch_frames = (
            video_reader.get_batch(batch_indices).asnumpy().astype("float32") / 255.0
        )
        left_video = torch.from_numpy(batch_frames).permute(0, 3, 1, 2).cuda()
        depth_tensor = (
            torch.from_numpy(video_depth[i : i + batch_size]).unsqueeze(1).cuda()
        )
        disparity = get_disparity(
            depth_tensor,
            max_disp,
            width,
            max_disp_reference_width,
        )

        with torch.inference_mode():
            right_video, occlusion_mask = stereo_projector(left_video, disparity)

        right_video = right_video.cpu().permute(0, 2, 3, 1).numpy()
        occlusion_mask = (
            occlusion_mask.cpu().permute(0, 2, 3, 1).numpy().repeat(3, axis=-1)
        )
        for j in range(len(batch_frames)):
            output_writer.write(
                np.concatenate([occlusion_mask[j], right_video[j]], axis=1)
            )

        del left_video, depth_tensor, disparity, right_video, occlusion_mask
        torch.cuda.empty_cache()
        gc.collect()

    output_writer.close()


if __name__ == "__main__":
    Fire(monitor_step("Step 1 - Depth Splatting")(main))
