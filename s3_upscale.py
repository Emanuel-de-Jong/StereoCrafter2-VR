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


def get_target_size(width, height, target_width, target_height):
    scale = min(target_width / width, target_height / height)
    target_width = make_even(width * scale)
    target_height = make_even(height * scale)

    return target_width, target_height


def main(
    input_video_path="outputs/vid_2_sbs.mp4",
    output_video_path="outputs/vid_3_upscale.mp4",
    video2x_path="dependency/Video2X/Video2X-x86_64.AppImage",
    target_width=5120,
    target_height=2560,
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

    if width >= target_width or height >= target_height:
        print(
            f"Skipping upscale because the video is already {target_width}px wide or {target_height}px high.",
            flush=True,
        )
        return

    output_width, output_height = get_target_size(
        width, height, target_width, target_height
    )
    print(f"Upscaling to: {output_width}x{output_height}", flush=True)

    os.makedirs(os.path.dirname(output_video_path) or ".", exist_ok=True)

    command = [
        video2x_path,
        "-i",
        input_video_path,
        "-o",
        output_video_path,
        "-p",
        "realesrgan",
        "-w",
        str(output_width),
        "-h",
        str(output_height),
        "--realesrgan-model",
        realesrgan_model,
    ]

    if gpu is not None:
        command.extend(["-g", str(gpu)])

    print("Running Video2X:", " ".join(command), flush=True)
    subprocess.run(command, check=True)


if __name__ == "__main__":
    Fire(main)
