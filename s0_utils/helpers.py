import gc
import subprocess
from pathlib import Path

import cv2
import numpy as np
import torch


def parse_bool(value):
    if isinstance(value, bool):
        return value
    if isinstance(value, int):
        return value != 0
    if isinstance(value, str):
        return value.strip().lower() in ["1", "true", "yes", "on"]
    return bool(value)


def parse_color(color, dtype=np.float32, normalize=True):
    values = [int(value.strip()) for value in color.split(",")]
    if len(values) != 3:
        raise ValueError(f"Expected color as R,G,B, got: {color}")
    if any(value < 0 or value > 255 for value in values):
        raise ValueError(f"Color values must be between 0 and 255, got: {color}")

    color_array = np.array(values, dtype=dtype)
    if normalize:
        color_array = color_array / 255.0
    return color_array


def get_video_properties(video):
    fps = video.get(cv2.CAP_PROP_FPS)
    width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))

    if fps <= 0:
        raise ValueError("Could not read video FPS")
    if width <= 0 or height <= 0:
        raise ValueError("Could not read video size")

    return fps, width, height


def get_video_fps(input_video_path):
    video = cv2.VideoCapture(str(input_video_path))
    if not video.isOpened():
        raise ValueError(f"Could not open video: {input_video_path}")

    fps = video.get(cv2.CAP_PROP_FPS)
    video.release()

    if fps <= 0:
        raise ValueError(f"Could not read video FPS: {input_video_path}")

    return fps


def get_video_size(input_video_path):
    video = cv2.VideoCapture(str(input_video_path))
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


def ensure_parent_dir(path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)


def get_input_basename(input_video_path):
    return Path(input_video_path).stem


def should_skip_output(output_path, overwrite=False):
    if Path(output_path).exists() and not parse_bool(overwrite):
        print(f"==> output already exists, skipping: {output_path}", flush=True)
        return True
    return False


def run_command(command):
    print("Running command:", " ".join(str(part) for part in command), flush=True)
    subprocess.run([str(part) for part in command], check=True)


def cleanup_cuda():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def format_duration(duration_seconds):
    duration_seconds = int(round(duration_seconds))
    hours = duration_seconds // 3600
    minutes = (duration_seconds % 3600) // 60
    seconds = duration_seconds % 60
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}"
