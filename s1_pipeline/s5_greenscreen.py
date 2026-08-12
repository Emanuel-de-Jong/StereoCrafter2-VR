import os
import shutil
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
from fire import Fire

sys.path.append(str(Path(__file__).resolve().parent.parent))

import s0_utils.global_params as g
from s0_utils.helpers import RawVideoWriter, parse_bool, should_skip_output
from s0_utils.monitor import monitor_step
from s0_utils.stereo import ForwardWarpStereo, get_disparity

GREEN = np.array([0.0, 1.0, 0.0], dtype=np.float32)


def main(
    input_video_path: str = str(g.OUTPUTS_DIR / "vid_4_upscale.mp4"),
    output_video_path: str = str(g.OUTPUTS_DIR / "vid_5_result.mp4"),
    depth_npz_path: str = str(g.OUTPUTS_DIR / "vid_1_splatting.npz"),
    enabled=True,
    rvm_model_path: str = "resnet50",
    rvm_downsample_ratio: float = 0.25,
    crf: int = 12,
    preset: str = "slow",
    overwrite: bool = False,
):
    enabled = parse_bool(enabled)

    if should_skip_output(output_video_path, overwrite):
        return

    if not os.path.isfile(input_video_path):
        raise FileNotFoundError(f"Input video not found: {input_video_path}")

    if not enabled:
        print("==> green screen disabled, copying input video", flush=True)
        shutil.copy2(input_video_path, output_video_path)
        return

    if rvm_downsample_ratio <= 0 or rvm_downsample_ratio > 1:
        raise ValueError(
            f"rvm_downsample_ratio must be greater than 0 and at most 1, got: {rvm_downsample_ratio}"
        )

    depth_maps, depth_fps, max_disp, max_disp_reference_width = load_depth_data(
        depth_npz_path
    )
    model, device = create_rvm_model(rvm_model_path)
    recurrent_state = [None] * 4
    stereo_projector = ForwardWarpStereo(return_occlusion_mask=False).to(device)

    video = cv2.VideoCapture(input_video_path)
    if not video.isOpened():
        raise ValueError(f"Could not open video: {input_video_path}")
    fps, width, height = get_video_properties(video)
    if width % 2 != 0:
        raise ValueError(f"SBS video width must be even, got: {width}")

    eye_width = width // 2
    os.makedirs(os.path.dirname(output_video_path) or ".", exist_ok=True)
    output_writer = RawVideoWriter(
        output_video_path,
        width,
        height,
        fps,
        codec="libx264",
        crf=crf,
        preset=preset,
        pixel_format="yuv420p",
    )

    frame_index = 0
    try:
        while True:
            success, frame_bgr = video.read()
            if not success:
                break

            frame_rgb = (
                cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
            )
            left_frame = frame_rgb[:, :eye_width]
            right_frame = frame_rgb[:, eye_width:]
            left_mask, recurrent_state = get_foreground_mask(
                model,
                left_frame,
                recurrent_state,
                device,
                rvm_downsample_ratio,
            )

            depth_index = min(
                round(frame_index * depth_fps / fps),
                len(depth_maps) - 1,
            )
            depth_frame = cv2.resize(
                depth_maps[depth_index],
                (eye_width, height),
                interpolation=cv2.INTER_LINEAR,
            )
            depth_tensor = (
                torch.from_numpy(depth_frame).unsqueeze(0).unsqueeze(0).to(device)
            )
            disparity = get_disparity(
                depth_tensor,
                max_disp,
                eye_width,
                max_disp_reference_width,
            )
            left_mask_tensor = (
                torch.from_numpy(left_mask).unsqueeze(0).unsqueeze(0).to(device)
            )
            with torch.inference_mode():
                right_mask_tensor = stereo_projector(
                    left_mask_tensor,
                    disparity,
                )
            right_mask = right_mask_tensor[0, 0].cpu().numpy()

            left_output = composite_green(left_frame, left_mask)
            right_output = composite_green(right_frame, right_mask)
            output_writer.write(np.concatenate([left_output, right_output], axis=1))

            del (
                depth_tensor,
                disparity,
                left_mask_tensor,
                right_mask_tensor,
            )
            frame_index += 1
            if frame_index % 25 == 0:
                print(f"==> green-screened {frame_index} frames", flush=True)
    finally:
        video.release()
        output_writer.close()

    print(f"==> saved green-screen video: {output_video_path}", flush=True)


def load_depth_data(depth_npz_path):
    if not os.path.isfile(depth_npz_path):
        raise FileNotFoundError(f"Depth map not found: {depth_npz_path}")
    depth_data = np.load(depth_npz_path)
    required_keys = ["depth", "fps", "max_disp", "max_disp_reference_width"]
    missing_keys = [key for key in required_keys if key not in depth_data]
    if missing_keys:
        raise ValueError(
            f"Depth npz is missing required arrays: {', '.join(missing_keys)}"
        )
    if len(depth_data["depth"]) <= 0:
        raise ValueError(f"Depth map has no frames: {depth_npz_path}")
    if float(depth_data["fps"].item()) <= 0:
        raise ValueError(f"Depth FPS must be positive: {depth_npz_path}")
    if int(depth_data["max_disp_reference_width"].item()) <= 0:
        raise ValueError(
            f"Depth disparity reference width must be positive: {depth_npz_path}"
        )
    return (
        depth_data["depth"].astype(np.float32),
        float(depth_data["fps"].item()),
        float(depth_data["max_disp"].item()),
        int(depth_data["max_disp_reference_width"].item()),
    )


def create_rvm_model(rvm_model_path):
    if rvm_model_path == "resnet50":
        model = torch.hub.load(
            "PeterL1n/RobustVideoMatting",
            "resnet50",
            trust_repo=True,
        )
    else:
        model = torch.hub.load(
            "PeterL1n/RobustVideoMatting",
            "resnet50",
            pretrained=False,
            trust_repo=True,
        )
        model.load_state_dict(torch.load(rvm_model_path, map_location="cpu"))

    device = torch.device("cuda")
    return model.to(device).eval(), device


def get_foreground_mask(
    model,
    frame_rgb,
    recurrent_state,
    device,
    downsample_ratio,
):
    frame_tensor = (
        torch.from_numpy(frame_rgb).permute(2, 0, 1).unsqueeze(0).float().to(device)
    )
    with torch.inference_mode():
        _foreground, alpha, *new_state = model(
            frame_tensor,
            *recurrent_state,
            downsample_ratio,
        )
    return alpha[0, 0].cpu().numpy(), new_state


def get_video_properties(video):
    fps = video.get(cv2.CAP_PROP_FPS)
    width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
    if fps <= 0:
        raise ValueError("Could not read video FPS")
    if width <= 0 or height <= 0:
        raise ValueError("Could not read video size")
    return fps, width, height


def composite_green(frame_rgb, mask):
    alpha = np.clip(mask, 0.0, 1.0).astype(np.float32)[:, :, None]
    return frame_rgb * alpha + GREEN.reshape(1, 1, 3) * (1.0 - alpha)


if __name__ == "__main__":
    Fire(monitor_step("Step 5 - Greenscreen")(main))
