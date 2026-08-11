import os
import math
import shutil
import sys
from pathlib import Path

import imageio_ffmpeg
from fire import Fire

sys.path.append(str(Path(__file__).resolve().parent.parent))

import s0_utils.global_params as g
from s0_utils.helpers import run_command, should_skip_output
from s0_utils.monitor import monitor_step


def main(
    input_video_path: str = str(g.OUTPUTS_DIR / "vid_3_greenscreen.mp4"),
    output_video_path: str = str(g.OUTPUTS_DIR / "vid_4_interp.mp4"),
    video2x_path: str = str(g.VIDEO2X_PATH),
    target_fps: int = 45,
    rife_model: str = "rife-v4.25",
    gpu: int = 0,
    scene_thresh: int = 100,
    crf: int = 16,
    preset: str = "medium",
    overwrite: bool = False,
):
    if should_skip_output(output_video_path, overwrite):
        return

    if not os.path.isfile(input_video_path):
        raise FileNotFoundError(f"Input video not found: {input_video_path}")
    if not os.path.isfile(video2x_path):
        raise FileNotFoundError(f"Video2X AppImage not found: {video2x_path}")
    if target_fps <= 0:
        raise ValueError(f"target_fps must be greater than 0, got: {target_fps}")

    os.makedirs(os.path.dirname(output_video_path) or ".", exist_ok=True)

    input_fps = get_video_fps(input_video_path)
    frame_rate_multiplier = get_frame_rate_multiplier(input_fps, target_fps)
    interpolated_fps = input_fps * frame_rate_multiplier
    print(f"Input FPS: {input_fps:.3f}", flush=True)
    print(f"Target FPS: {target_fps}", flush=True)

    if input_fps >= target_fps:
        print("Input FPS is already at or above target FPS, copying video.", flush=True)
        shutil.copy2(input_video_path, output_video_path)
        return

    print(f"RIFE model: {rife_model}", flush=True)
    print(f"Frame rate multiplier: {frame_rate_multiplier}x", flush=True)

    if abs(interpolated_fps - target_fps) < 0.01:
        run_rife_interpolation(
            video2x_path,
            input_video_path,
            output_video_path,
            frame_rate_multiplier,
            rife_model,
            gpu,
            scene_thresh,
        )
        return

    temp_output_path = os.path.join(
        os.path.dirname(output_video_path) or ".",
        f".{os.path.splitext(os.path.basename(output_video_path))[0]}_rife.mp4",
    )

    try:
        run_rife_interpolation(
            video2x_path,
            input_video_path,
            temp_output_path,
            frame_rate_multiplier,
            rife_model,
            gpu,
            scene_thresh,
        )
        conform_video_fps(
            temp_output_path,
            output_video_path,
            target_fps,
            crf,
            preset,
        )
    finally:
        if os.path.exists(temp_output_path):
            os.remove(temp_output_path)


def get_video_fps(input_video_path):
    import cv2

    video = cv2.VideoCapture(str(input_video_path))
    if not video.isOpened():
        raise ValueError(f"Could not open video: {input_video_path}")

    fps = video.get(cv2.CAP_PROP_FPS)
    video.release()

    if fps <= 0:
        raise ValueError(f"Could not read video FPS: {input_video_path}")

    return fps


def get_frame_rate_multiplier(input_fps, target_fps):
    return max(1, int(math.ceil(target_fps / input_fps)))


def run_rife_interpolation(
    video2x_path,
    input_video_path,
    output_video_path,
    frame_rate_multiplier,
    rife_model,
    gpu,
    scene_thresh,
):
    command = [
        video2x_path,
        "-i",
        input_video_path,
        "-o",
        output_video_path,
        "-m",
        str(frame_rate_multiplier),
        "-p",
        "rife",
        "--rife-model",
        rife_model,
        "-t",
        str(scene_thresh),
    ]

    if gpu is not None:
        command.extend(["-d", str(gpu)])

    print("Running RIFE interpolation", flush=True)
    run_command(command)


def conform_video_fps(input_video_path, output_video_path, target_fps, crf, preset):
    ffmpeg_path = imageio_ffmpeg.get_ffmpeg_exe()
    command = [
        ffmpeg_path,
        "-y",
        "-i",
        input_video_path,
        "-vf",
        f"fps={target_fps}",
        "-c:v",
        "libx264",
        "-crf",
        str(crf),
        "-preset",
        preset,
        "-pix_fmt",
        "yuv420p",
        "-c:a",
        "copy",
        output_video_path,
    ]

    print("Conforming FPS", flush=True)
    run_command(command)


if __name__ == "__main__":
    Fire(monitor_step("Step 4 - Interpolation")(main))
