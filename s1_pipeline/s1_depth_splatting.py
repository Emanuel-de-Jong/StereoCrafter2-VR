import gc
import cv2
import os
import sys
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.append(str(Path(__file__).resolve().parent.parent))

from diffusers.training_utils import set_seed
from fire import Fire
from decord import VideoReader, cpu

import s0_utils.global_params as g
from s0_utils.helpers import RawVideoWriter, cleanup_cuda, should_skip_output
from s0_utils.monitor import monitor_step

from dependencies.DepthCrafter.depthcrafter.depth_crafter_ppl import (
    DepthCrafterPipeline,
)
from dependencies.DepthCrafter.depthcrafter.unet import (
    DiffusersUNetSpatioTemporalConditionModelDepthCrafter,
)
from dependencies.DepthCrafter.depthcrafter.utils import vis_sequence_depth

from Forward_Warp import forward_warp


def main(
    input_video_path: str = str(g.INPUTS_DIR / "vid.mp4"),
    output_video_path: str = str(g.OUTPUTS_DIR / "vid_1_splatting.mkv"),
    unet_path: str = str(g.DEPTHCRAFTER_WEIGHTS_PATH),
    pre_trained_path: str = str(g.SVD_WEIGHTS_PATH),
    max_disp: float = 26,
    max_disp_reference_width: int = 1920,
    process_length: int = -1,
    batch_size: int = 10,
    cpu_offload: str = "model",
    num_denoising_steps: int = 8,
    guidance_scale: float = 1.2,
    window_size: int = 70,
    overlap: int = 25,
    max_res: int = 1024,
    dataset: str = "open",
    target_fps: int = 30,
    seed: int = 42,
    track_time: bool = False,
    save_depth: bool = True,
    decode_chunk_size: int = 8,
    depth_low_percentile: float = 1.0,
    depth_high_percentile: float = 99.0,
    scene_cut_threshold: float = 0.22,
    mask_dilation: int = 4,
    depth_edge_threshold: float = 0.03,
    overwrite: bool = False,
):
    if should_skip_output(output_video_path, overwrite):
        return

    depthcrafter_demo = DepthCrafterDemo(
        unet_path=unet_path,
        pre_trained_path=pre_trained_path,
        cpu_offload=cpu_offload,
    )

    video_depth = depthcrafter_demo.infer(
        input_video_path=input_video_path,
        output_video_path=output_video_path,
        process_length=process_length,
        num_denoising_steps=num_denoising_steps,
        guidance_scale=guidance_scale,
        window_size=window_size,
        overlap=overlap,
        max_res=max_res,
        dataset=dataset,
        target_fps=target_fps,
        seed=seed,
        track_time=track_time,
        save_depth=save_depth,
        decode_chunk_size=decode_chunk_size,
        depth_low_percentile=depth_low_percentile,
        depth_high_percentile=depth_high_percentile,
        scene_cut_threshold=scene_cut_threshold,
    )

    print("==> unloading DepthCrafter before splatting", flush=True)
    del depthcrafter_demo
    cleanup_cuda()

    print("==> running depth-based forward splatting", flush=True)
    DepthSplatting(
        input_video_path,
        output_video_path,
        video_depth,
        max_disp,
        max_disp_reference_width,
        process_length,
        batch_size,
        target_fps,
        mask_dilation,
        depth_edge_threshold,
    )


def read_video_frames(video_path, process_length, target_fps, max_res, dataset="open"):
    if dataset == "open":
        print("==> processing video: ", video_path, flush=True)
        vid = VideoReader(video_path, ctx=cpu(0))
        print(
            "==> original video shape: ",
            (len(vid), *vid.get_batch([0]).shape[1:]),
            flush=True,
        )
        original_height, original_width = vid.get_batch([0]).shape[1:3]
        height = round(original_height / 64) * 64
        width = round(original_width / 64) * 64
        if max(height, width) > max_res:
            scale = max_res / max(original_height, original_width)
            height = round(original_height * scale / 64) * 64
            width = round(original_width * scale / 64) * 64
    else:
        raise ValueError(f"Unknown dataset: {dataset}")

    vid = VideoReader(video_path, ctx=cpu(0), width=width, height=height)

    avg_fps = vid.get_avg_fps()
    fps = avg_fps if target_fps == -1 else min(target_fps, avg_fps)
    stride = round(avg_fps / fps)
    stride = max(stride, 1)
    frames_idx = list(range(0, len(vid), stride))
    print(
        f"==> downsampled shape: {len(frames_idx), *vid.get_batch([0]).shape[1:]}, with stride: {stride}",
        flush=True,
    )
    if process_length != -1 and process_length < len(frames_idx):
        frames_idx = frames_idx[:process_length]
    print(
        f"==> final processing shape: {len(frames_idx), *vid.get_batch([0]).shape[1:]}",
        flush=True,
    )
    frames = vid.get_batch(frames_idx).asnumpy().astype("float32") / 255.0

    return frames, fps, original_height, original_width


class DepthCrafterDemo:
    def __init__(
        self,
        unet_path: str,
        pre_trained_path: str,
        cpu_offload: str = "model",
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

        if isinstance(cpu_offload, str):
            cpu_offload = cpu_offload.lower()
            if cpu_offload in ["none", "cuda", "false"]:
                cpu_offload = None

        if cpu_offload is not None:
            if cpu_offload == "sequential":
                self.pipe.enable_sequential_cpu_offload()
            elif cpu_offload == "model":
                self.pipe.enable_model_cpu_offload()
            else:
                raise ValueError(f"Unknown cpu offload option: {cpu_offload}")
        else:
            self.pipe.to("cuda")
        try:
            self.pipe.enable_xformers_memory_efficient_attention()
        except Exception as e:
            print(e, flush=True)
            print("Xformers is not enabled", flush=True)
        self.pipe.enable_attention_slicing()

    def infer(
        self,
        input_video_path: str,
        output_video_path: str,
        process_length: int = -1,
        num_denoising_steps: int = 8,
        guidance_scale: float = 1.2,
        window_size: int = 70,
        overlap: int = 25,
        max_res: int = 1024,
        dataset: str = "open",
        target_fps: int = -1,
        seed: int = 42,
        track_time: bool = False,
        save_depth: bool = True,
        decode_chunk_size: int = 8,
        depth_low_percentile: float = 1.0,
        depth_high_percentile: float = 99.0,
        scene_cut_threshold: float = 0.22,
    ):
        set_seed(seed)

        print("==> loading video frames", flush=True)
        frames, target_fps, original_height, original_width = read_video_frames(
            input_video_path,
            process_length,
            target_fps,
            max_res,
            dataset,
        )

        scene_ranges = detect_scene_ranges(frames, scene_cut_threshold)
        print(
            f"==> running DepthCrafter depth inference across {len(scene_ranges)} scene(s)",
            flush=True,
        )
        scene_depths = []
        with torch.inference_mode():
            for scene_index, (scene_start, scene_end) in enumerate(scene_ranges):
                print(
                    f"==> depth scene {scene_index + 1}/{len(scene_ranges)}: frames {scene_start + 1}-{scene_end}",
                    flush=True,
                )
                scene_depth = self.pipe(
                    frames[scene_start:scene_end],
                    height=frames.shape[1],
                    width=frames.shape[2],
                    output_type="np",
                    guidance_scale=guidance_scale,
                    num_inference_steps=num_denoising_steps,
                    window_size=min(window_size, scene_end - scene_start),
                    overlap=min(overlap, max(scene_end - scene_start - 1, 0)),
                    track_time=track_time,
                    decode_chunk_size=decode_chunk_size,
                ).frames[0]
                scene_depths.append(scene_depth.sum(-1) / scene_depth.shape[-1])

        print("==> post-processing depth maps", flush=True)
        res = np.concatenate(scene_depths, axis=0)

        resized_res = []
        for i in range(0, len(res), decode_chunk_size):
            tensor_res = (
                torch.tensor(res[i : i + decode_chunk_size])
                .unsqueeze(1)
                .float()
                .contiguous()
                .cuda()
            )
            tensor_res = F.interpolate(
                tensor_res,
                size=(original_height, original_width),
                mode="bilinear",
                align_corners=False,
            )
            resized_res.append(tensor_res.cpu().numpy()[:, 0, :, :])
            del tensor_res
        res = np.concatenate(resized_res, axis=0)

        res = normalize_depth_scenes(
            res,
            scene_ranges,
            depth_low_percentile,
            depth_high_percentile,
        )
        vis = vis_sequence_depth(res)
        save_path = os.path.join(
            os.path.dirname(output_video_path),
            os.path.splitext(os.path.basename(output_video_path))[0],
        )

        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        if save_depth:
            np.savez_compressed(save_path + ".npz", depth=res)
            write_rgb_video(save_path + "_depth_vis.mp4", vis, target_fps)

        return res


def detect_scene_ranges(frames, threshold):
    if len(frames) <= 1 or threshold <= 0:
        return [(0, len(frames))]

    scene_starts = [0]
    previous_frame = cv2.resize(frames[0], (64, 64), interpolation=cv2.INTER_AREA)
    previous_gray = cv2.cvtColor(previous_frame, cv2.COLOR_RGB2GRAY) * 255.0

    for frame_index in range(1, len(frames)):
        current_frame = cv2.resize(
            frames[frame_index], (64, 64), interpolation=cv2.INTER_AREA
        )
        current_gray = cv2.cvtColor(current_frame, cv2.COLOR_RGB2GRAY) * 255.0
        frame_difference = float(np.mean(np.abs(current_gray - previous_gray))) / 255.0
        if frame_difference >= threshold:
            scene_starts.append(frame_index)
        previous_gray = current_gray

    scene_starts.append(len(frames))
    return [
        (scene_starts[index], scene_starts[index + 1])
        for index in range(len(scene_starts) - 1)
    ]


def normalize_depth_scenes(depth, scene_ranges, low_percentile, high_percentile):
    normalized_depth = np.empty_like(depth, dtype=np.float32)

    for scene_start, scene_end in scene_ranges:
        scene_depth = depth[scene_start:scene_end]
        depth_low = float(np.percentile(scene_depth, low_percentile))
        depth_high = float(np.percentile(scene_depth, high_percentile))
        depth_range = max(depth_high - depth_low, 1e-6)
        normalized_depth[scene_start:scene_end] = np.clip(
            (scene_depth - depth_low) / depth_range,
            0.0,
            1.0,
        )

    return normalized_depth


def write_rgb_video(video_path, frames, fps):
    frames_uint8 = np.clip(frames, 0.0, 255.0)
    if frames_uint8.max() <= 1.0:
        frames_uint8 = frames_uint8 * 255.0
    frames_uint8 = frames_uint8.astype(np.uint8)

    height, width = frames_uint8.shape[1:3]
    video_writer = cv2.VideoWriter(
        video_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height)
    )

    for frame in frames_uint8:
        video_writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))

    video_writer.release()


class ForwardWarpStereo(nn.Module):
    def __init__(self, eps=1e-6, occlu_map=False):
        super(ForwardWarpStereo, self).__init__()
        self.eps = eps
        self.occlu_map = occlu_map
        self.fw = forward_warp()

    def forward(self, im, disp):
        im = im.contiguous()
        disp = disp.contiguous()
        weights_map = disp - disp.min()
        weights_map = (1.414) ** weights_map
        flow = -disp.squeeze(1)
        dummy_flow = torch.zeros_like(flow, requires_grad=False)
        flow = torch.stack((flow, dummy_flow), dim=-1)
        res_accum = self.fw(im * weights_map, flow)
        mask = self.fw(weights_map, flow)
        mask.clamp_(min=self.eps)
        res = res_accum / mask
        if not self.occlu_map:
            return res
        else:
            ones = torch.ones_like(disp, requires_grad=False)
            occlu_map = self.fw(ones, flow)
            occlu_map.clamp_(0.0, 1.0)
            occlu_map = 1.0 - occlu_map
            return res, occlu_map


def DepthSplatting(
    input_video_path,
    output_video_path,
    video_depth,
    max_disp,
    max_disp_reference_width,
    process_length,
    batch_size,
    target_fps,
    mask_dilation,
    depth_edge_threshold,
):
    print("==> loading frames for splatting", flush=True)
    vid_reader = VideoReader(input_video_path, ctx=cpu(0))
    original_fps = vid_reader.get_avg_fps()
    fps = original_fps if target_fps == -1 else target_fps
    stride = round(original_fps / fps)
    stride = max(stride, 1)
    frames_idx = list(range(0, len(vid_reader), stride))

    if process_length != -1 and process_length < len(frames_idx):
        frames_idx = frames_idx[:process_length]

    video_depth = video_depth[: len(frames_idx)]

    stereo_projector = ForwardWarpStereo(occlu_map=True).cuda()

    num_frames = len(frames_idx)
    height, width = vid_reader.get_batch([frames_idx[0]]).shape[1:3]

    effective_max_disp = max_disp * width / max_disp_reference_width
    print(f"==> effective maximum disparity: {effective_max_disp:.2f}px", flush=True)
    out = RawVideoWriter(
        output_video_path,
        width * 2,
        height,
        fps,
        codec="ffv1",
    )

    for i in range(0, num_frames, batch_size):
        print(
            f"==> splatting frames {i + 1}-{min(i + batch_size, num_frames)} / {num_frames}",
            flush=True,
        )
        batch_indices = frames_idx[i : i + batch_size]
        batch_frames = (
            vid_reader.get_batch(batch_indices).asnumpy().astype("float32") / 255.0
        )
        batch_depth = video_depth[i : i + batch_size]
        left_video = torch.from_numpy(batch_frames).permute(0, 3, 1, 2).float().cuda()
        disp_map = torch.from_numpy(batch_depth).unsqueeze(1).float().cuda()

        disp_map = disp_map * 2.0 - 1.0
        disp_map = disp_map * effective_max_disp

        depth_edges_x = torch.abs(disp_map[:, :, :, 1:] - disp_map[:, :, :, :-1])
        depth_edges_y = torch.abs(disp_map[:, :, 1:, :] - disp_map[:, :, :-1, :])
        depth_edges = torch.zeros_like(disp_map)
        depth_edges[:, :, :, 1:] = torch.maximum(
            depth_edges[:, :, :, 1:], depth_edges_x
        )
        depth_edges[:, :, 1:, :] = torch.maximum(
            depth_edges[:, :, 1:, :], depth_edges_y
        )
        depth_edges = (depth_edges >= effective_max_disp * depth_edge_threshold).float()

        with torch.no_grad():
            right_video, occlusion_mask = stereo_projector(left_video, disp_map)
            warped_depth_edges = stereo_projector(depth_edges, disp_map)[0]

        repair_mask = torch.maximum(occlusion_mask, (warped_depth_edges > 0.05).float())
        if mask_dilation > 0:
            kernel_size = mask_dilation * 2 + 1
            repair_mask = F.max_pool2d(
                repair_mask,
                kernel_size=kernel_size,
                stride=1,
                padding=mask_dilation,
            )

        right_video = right_video.cpu().permute(0, 2, 3, 1).numpy()
        repair_mask = repair_mask.cpu().permute(0, 2, 3, 1).numpy().repeat(3, axis=-1)

        for j in range(len(batch_frames)):
            condition_frame = np.concatenate([repair_mask[j], right_video[j]], axis=1)
            out.write(condition_frame)

        del (
            left_video,
            disp_map,
            depth_edges,
            depth_edges_x,
            depth_edges_y,
            warped_depth_edges,
            right_video,
            occlusion_mask,
            repair_mask,
        )
        torch.cuda.empty_cache()
        gc.collect()

    out.close()


if __name__ == "__main__":
    Fire(monitor_step("Step 1 - Depth Splatting")(main))
