import os
import shutil
import sys
import tempfile
from pathlib import Path

import cv2
import imageio_ffmpeg
import numpy as np
from fire import Fire

sys.path.append(str(Path(__file__).resolve().parent.parent))

import s0_utils.global_params as g
from s0_utils.helpers import (
    parse_bool,
    run_command,
    should_skip_output,
)
from s0_utils.monitor import monitor_step


def main(
    input_video_path: str = str(g.OUTPUTS_DIR / "vid_5_upscale.mp4"),
    output_video_path: str = str(g.OUTPUTS_DIR / "vid_6_result.mp4"),
    enabled: bool = True,
    green: str = "0,255,0",
    rgb_tolerance: float = 48.0,
    green_dominance: float = 32.0,
    write_metadata: bool = True,
    metadata_items: str = "",
    overwrite: bool = False,
):
    enabled = parse_bool(enabled)
    write_metadata = parse_bool(write_metadata)
    overwrite = parse_bool(overwrite)

    if should_skip_output(output_video_path, overwrite):
        return

    if not os.path.isfile(input_video_path):
        raise FileNotFoundError(f"Input video not found: {input_video_path}")

    os.makedirs(os.path.dirname(output_video_path) or ".", exist_ok=True)

    if not enabled:
        print("==> green cleanup disabled, copying input video", flush=True)
        shutil.copy2(input_video_path, output_video_path)
        return

    green_color = parse_color(green, dtype=np.uint8, normalize=False)
    video = cv2.VideoCapture(input_video_path)
    if not video.isOpened():
        raise ValueError(f"Could not open video: {input_video_path}")

    fps, width, height = get_video_properties(video)
    writer = cv2.VideoWriter(
        output_video_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height)
    )

    frame_index = 0

    try:
        while True:
            success, frame_bgr = video.read()
            if not success:
                break

            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            cleanup_mask = get_green_cleanup_mask(
                frame_rgb,
                green_color,
                rgb_tolerance,
                green_dominance,
            )
            frame_rgb[cleanup_mask] = green_color

            writer.write(cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR))

            frame_index += 1
            if frame_index % 100 == 0:
                print(f"==> cleaned {frame_index} frames", flush=True)
    finally:
        video.release()
        writer.release()

    if write_metadata:
        print("==> writing chroma-key metadata", flush=True)
        write_mp4_metadata(output_video_path, green_color, metadata_items)

    print(f"==> saved green-cleaned video: {output_video_path}", flush=True)


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


def get_green_cleanup_mask(frame_rgb, green_color, rgb_tolerance, green_dominance):
    frame_float = frame_rgb.astype(np.float32)
    green_float = green_color.astype(np.float32).reshape(1, 1, 3)
    distance = np.linalg.norm(frame_float - green_float, axis=2)
    green_channel = frame_float[:, :, 1]
    red_channel = frame_float[:, :, 0]
    blue_channel = frame_float[:, :, 2]

    close_to_green = distance <= rgb_tolerance
    green_dominant = green_channel >= red_channel + green_dominance
    green_dominant &= green_channel >= blue_channel + green_dominance

    return close_to_green & green_dominant


def get_hex_color(green_color):
    return "#{:02X}{:02X}{:02X}".format(
        int(green_color[0]), int(green_color[1]), int(green_color[2])
    )


def parse_metadata_items(metadata_items):
    if not metadata_items:
        return []

    items = []
    for metadata_item in metadata_items.split(","):
        metadata_item = metadata_item.strip()
        if not metadata_item:
            continue
        if "=" not in metadata_item:
            raise ValueError(
                f"Expected metadata item as key=value, got: {metadata_item}"
            )
        key, value = metadata_item.split("=", 1)
        items.append((key.strip(), value.strip()))
    return items


def get_chroma_key_metadata(green_color, metadata_items):
    hex_color = get_hex_color(green_color)
    metadata = [
        ("stereo_mode", "left_right"),
        ("chroma_key", "true"),
        ("chroma_key_color", hex_color),
        ("greenscreen", "true"),
        ("greenscreen_color", hex_color),
        ("passthrough_chroma_key", hex_color),
        ("com.oculus.vr.chroma_key", hex_color),
        ("com.meta.vr.chroma_key", hex_color),
        ("com.deovr.chroma_key", hex_color),
    ]
    metadata.extend(parse_metadata_items(metadata_items))
    return metadata


def write_mp4_metadata(video_path, green_color, metadata_items):
    metadata = get_chroma_key_metadata(green_color, metadata_items)
    ffmpeg_path = imageio_ffmpeg.get_ffmpeg_exe()
    directory = os.path.dirname(video_path) or "."
    file_name = os.path.basename(video_path)

    with tempfile.NamedTemporaryFile(
        prefix=f".{os.path.splitext(file_name)[0]}_metadata_",
        suffix=".mp4",
        dir=directory,
        delete=False,
    ) as temp_file:
        temp_output_path = temp_file.name

    command = [
        ffmpeg_path,
        "-y",
        "-i",
        video_path,
        "-map",
        "0",
        "-c",
        "copy",
        "-movflags",
        "use_metadata_tags",
    ]

    for key, value in metadata:
        command.extend(["-metadata", f"{key}={value}"])

    command.append(temp_output_path)

    try:
        run_command(command)
        os.replace(temp_output_path, video_path)
    finally:
        if os.path.exists(temp_output_path):
            os.remove(temp_output_path)


if __name__ == "__main__":
    Fire(monitor_step("Step 6 - Green Cleanup")(main))
