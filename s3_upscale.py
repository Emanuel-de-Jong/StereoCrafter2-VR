import os
import subprocess

import cv2
from fire import Fire


def get_video_size(input_video_path):
    video = cv2.VideoCapture(input_video_path)
    if not video.isOpened():
        raise ValueError(f"Could not open video: {input_video_path}")

    width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
    video.release()

    if width <= 0 or height <= 0:
        raise ValueError(f"Could not read video size: {input_video_path}")

    return width, height


def make_even(value):
    value = int(round(value))
    return value if value % 2 == 0 else value + 1


def get_target_size(width, height, target_size):
    scale = target_size / max(width, height)
    target_width = make_even(width * scale)
    target_height = make_even(height * scale)

    if width >= height:
        target_width = target_size
        target_height = make_even(height * scale)
    else:
        target_width = make_even(width * scale)
        target_height = target_size

    return target_width, target_height


def main(
    input_video_path="outputs/vid_2_sbs.mp4",
    output_video_path="outputs/vid_3_upscale.mp4",
    video2x_path="dependency/Video2X/Video2X-x86_64.AppImage",
    target_size=2560,
    realesrgan_model="realesr-animevideov3",
    gpu=0,
    overwrite=False,
):
    if os.path.exists(output_video_path) and not overwrite:
        print(f"==> output already exists, skipping: {output_video_path}", flush=True)
        return

    if not os.path.isfile(input_video_path):
        raise FileNotFoundError(f"Input video not found: {input_video_path}")
    if not os.path.isfile(video2x_path):
        raise FileNotFoundError(f"Video2X AppImage not found: {video2x_path}")

    width, height = get_video_size(input_video_path)
    print(f"Input video size: {width}x{height}", flush=True)

    if max(width, height) >= target_size:
        print(f"Skipping upscale because one dimension is already {target_size}px or larger.", flush=True)
        return

    target_width, target_height = get_target_size(width, height, target_size)
    print(f"Upscaling to: {target_width}x{target_height}", flush=True)

    os.makedirs(os.path.dirname(output_video_path) or ".", exist_ok=True)

    command = [
        video2x_path,
        "-i", input_video_path,
        "-o", output_video_path,
        "-p", "realesrgan",
        "-w", str(target_width),
        "-h", str(target_height),
        "--realesrgan-model", realesrgan_model,
    ]

    if gpu is not None:
        command.extend(["-g", str(gpu)])

    print("Running Video2X:", " ".join(command), flush=True)
    subprocess.run(command, check=True)


if __name__ == "__main__":
    Fire(main)
