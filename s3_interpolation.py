import os
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


def run_interpolation(input_video_path, output_video_path, target_fps, crf, preset):
    ffmpeg_path = imageio_ffmpeg.get_ffmpeg_exe()
    command = [
        ffmpeg_path,
        "-y",
        "-i",
        input_video_path,
        "-vf",
        f"minterpolate=fps={target_fps}:mi_mode=mci:mc_mode=aobmc:me_mode=bidir:vsbmc=1",
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

    print("Running interpolation:", " ".join(command), flush=True)
    subprocess.run(command, check=True)


def main(
    input_video_path="outputs/vid_2_sbs.mp4",
    output_video_path="outputs/vid_3_interp.mp4",
    target_fps=30,
    crf=16,
    preset="medium",
    overwrite=False,
):
    if os.path.exists(output_video_path) and not overwrite:
        print(f"==> output already exists, skipping: {output_video_path}", flush=True)
        return

    if not os.path.isfile(input_video_path):
        raise FileNotFoundError(f"Input video not found: {input_video_path}")
    if target_fps <= 0:
        raise ValueError(f"target_fps must be greater than 0, got: {target_fps}")

    os.makedirs(os.path.dirname(output_video_path) or ".", exist_ok=True)

    input_fps = get_video_fps(input_video_path)
    print(f"Input FPS: {input_fps:.3f}", flush=True)
    print(f"Target FPS: {target_fps}", flush=True)

    if input_fps >= target_fps:
        print("Input FPS is already at or above target FPS, copying video.", flush=True)
        shutil.copy2(input_video_path, output_video_path)
        return

    run_interpolation(input_video_path, output_video_path, target_fps, crf, preset)


if __name__ == "__main__":
    Fire(main)
