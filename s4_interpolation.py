import os
import math
import shutil
import subprocess

import cv2
import imageio_ffmpeg
from fire import Fire


def get_video_fps(input_video_path):
    video = cv2.VideoCapture(input_video_path)
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

    print("Running RIFE interpolation:", " ".join(command), flush=True)
    subprocess.run(command, check=True)


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

    print("Conforming FPS:", " ".join(command), flush=True)
    subprocess.run(command, check=True)


def main(
    input_video_path="outputs/vid_2_sbs.mp4",
    output_video_path="outputs/vid_3_interp.mp4",
    video2x_path="dependency/Video2X/Video2X-x86_64.AppImage",
    target_fps=30,
    rife_model="rife-v4.25",
    gpu=0,
    scene_thresh=100,
    crf=16,
    preset="medium",
    overwrite=False,
):
    if os.path.exists(output_video_path) and not overwrite:
        print(f"==> output already exists, skipping: {output_video_path}", flush=True)
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
        conform_video_fps(temp_output_path, output_video_path, target_fps, crf, preset)
    finally:
        if os.path.exists(temp_output_path):
            os.remove(temp_output_path)


if __name__ == "__main__":
    Fire(main)
