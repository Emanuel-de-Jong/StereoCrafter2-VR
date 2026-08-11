import os
import sys
from pathlib import Path

import imageio_ffmpeg
import cv2
from fire import Fire

sys.path.append(str(Path(__file__).resolve().parent.parent))

import s0_utils.global_params as g
from s0_utils.helpers import run_command, should_skip_output
from s0_utils.monitor import monitor_step


def main(
    input_video_path: str = str(g.OUTPUTS_DIR / "vid_4_interp.mp4"),
    output_video_path: str = str(g.OUTPUTS_DIR / "vid_5_upscale.mp4"),
    video2x_path: str = str(g.VIDEO2X_PATH),
    target_width: int = 5120,
    target_height: int = 2560,
    realesrgan_model: str = "realesr-animevideov3",
    gpu: int = 0,
    overwrite: bool = False,
):
    if should_skip_output(output_video_path, overwrite):
        return

    if not os.path.isfile(input_video_path):
        raise FileNotFoundError(f"Input video not found: {input_video_path}")
    if not os.path.isfile(video2x_path):
        raise FileNotFoundError(f"Video2X AppImage not found: {video2x_path}")

    width, height = get_video_size(input_video_path)
    print(f"Input video size: {width}x{height}", flush=True)

    target_output_width, target_output_height = get_target_size(
        width, height, target_width, target_height
    )
    print(
        f"Target output size: {target_output_width}x{target_output_height}", flush=True
    )

    os.makedirs(os.path.dirname(output_video_path) or ".", exist_ok=True)

    current_input_path = input_video_path
    current_width = width
    current_height = height
    temp_paths = []
    pass_index = 0

    while True:
        scaling_factor = get_realesrgan_scaling_factor(
            current_width, current_height, target_output_width, target_output_height
        )

        if scaling_factor is None:
            break

        pass_index += 1
        temp_output_path = os.path.join(
            os.path.dirname(output_video_path) or ".",
            f".{os.path.splitext(os.path.basename(output_video_path))[0]}_realesrgan_{pass_index}.mp4",
        )
        temp_paths.append(temp_output_path)

        output_width = current_width * scaling_factor
        output_height = current_height * scaling_factor
        print(
            f"RealESRGAN pass {pass_index}: {current_width}x{current_height} -> {output_width}x{output_height} ({scaling_factor}x)",
            flush=True,
        )
        run_video2x(
            video2x_path,
            current_input_path,
            temp_output_path,
            scaling_factor,
            realesrgan_model,
            gpu,
        )

        current_input_path = temp_output_path
        current_width = output_width
        current_height = output_height

    if current_width == target_output_width and current_height == target_output_height:
        if current_input_path == input_video_path:
            os.makedirs(os.path.dirname(output_video_path) or ".", exist_ok=True)
            resize_video(
                current_input_path,
                output_video_path,
                current_width,
                current_height,
            )
        else:
            os.replace(current_input_path, output_video_path)
    else:
        print(
            f"Exact resize: {current_width}x{current_height} -> {target_output_width}x{target_output_height}",
            flush=True,
        )
        resize_video(
            current_input_path,
            output_video_path,
            target_output_width,
            target_output_height,
        )

    for temp_path in temp_paths:
        if os.path.exists(temp_path) and temp_path != output_video_path:
            os.remove(temp_path)


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


def get_target_size(width, height, target_width, target_height):
    scale = min(target_width / width, target_height / height)
    target_width = make_even(width * scale)
    target_height = make_even(height * scale)

    return target_width, target_height


def get_realesrgan_scaling_factor(width, height, target_width, target_height):
    for scaling_factor in [4, 3, 2]:
        if (
            width * scaling_factor <= target_width
            and height * scaling_factor <= target_height
        ):
            return scaling_factor

    return None


def run_video2x(
    video2x_path,
    input_video_path,
    output_video_path,
    scaling_factor,
    realesrgan_model,
    gpu,
):
    command = [
        video2x_path,
        "-i",
        input_video_path,
        "-o",
        output_video_path,
        "-p",
        "realesrgan",
        "-s",
        str(scaling_factor),
        "--realesrgan-model",
        realesrgan_model,
    ]

    if gpu is not None:
        command.extend(["-d", str(gpu)])

    print("Running Video2X", flush=True)
    run_command(command)


def resize_video(input_video_path, output_video_path, width, height):
    ffmpeg_path = imageio_ffmpeg.get_ffmpeg_exe()
    command = [
        ffmpeg_path,
        "-y",
        "-i",
        input_video_path,
        "-vf",
        f"scale={width}:{height}:flags=lanczos",
        "-c:v",
        "libx264",
        "-pix_fmt",
        "yuv420p",
        "-c:a",
        "copy",
        output_video_path,
    ]

    print("Running resize", flush=True)
    run_command(command)


if __name__ == "__main__":
    Fire(monitor_step("Step 5 - Upscale")(main))
