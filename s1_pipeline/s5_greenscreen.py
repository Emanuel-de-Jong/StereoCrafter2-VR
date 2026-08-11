import json
import os
import shutil
import sys
import tempfile
from pathlib import Path

import cv2
import imageio_ffmpeg
import numpy as np
import torch
from fire import Fire
from PIL import Image

sys.path.append(str(Path(__file__).resolve().parent.parent))

import s0_utils.global_params as g
from s0_utils.helpers import (
    RawVideoWriter,
    parse_bool,
    run_command,
    should_skip_output,
)
from s0_utils.monitor import monitor_step


def main(
    input_video_path=str(g.OUTPUTS_DIR / "vid_4_upscale.mp4"),
    output_video_path=str(g.OUTPUTS_DIR / "vid_5_result.mp4"),
    depth_npz_path=str(g.OUTPUTS_DIR / "vid_1_splatting.npz"),
    enabled=True,
    model_type="detr",
    model_path="facebook/detr-resnet-50-panoptic",
    rvm_model_path="resnet50",
    rvm_downsample_ratio=0.25,
    foreground_classes="person",
    selection_mode="main_subject",
    green="0,255,0",
    threshold=0.5,
    main_subject_score_ratio=0.65,
    min_segment_area=0.01,
    max_segment_area=0.85,
    close_percentile=80.0,
    min_depth_component_area=0.005,
    mask_feather=15,
    mask_erode=3,
    mask_dilate=3,
    temporal_smoothing=0.3,
    temporal_smoothing_reference_fps=30.0,
    stereo_mask_mode="depth",
    max_disp=26,
    max_disp_reference_width=1920,
    convergence=0.5,
    source_fps=30.0,
    write_metadata=True,
    metadata_items="",
    crf=12,
    preset="slow",
    device="cuda",
    overwrite=False,
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
        print("==> green screen disabled, copying input video", flush=True)
        shutil.copy2(input_video_path, output_video_path)
        return

    model_type = model_type.lower()
    if model_type not in ["detr", "rvm"]:
        raise ValueError(f"Unknown model_type: {model_type}")
    stereo_mask_mode = stereo_mask_mode.lower()
    if stereo_mask_mode not in ["left", "both", "union", "depth"]:
        raise ValueError(f"Unknown stereo_mask_mode: {stereo_mask_mode}")
    selection_mode = selection_mode.lower()
    if selection_mode not in ["main_subject", "classes", "all_segments"]:
        raise ValueError(f"Unknown selection_mode: {selection_mode}")

    green_color = parse_color(green)
    convergence = load_convergence(depth_npz_path, convergence)
    foreground_class_set = parse_classes(foreground_classes)
    depth_maps = load_depth_maps(depth_npz_path)

    if model_type == "rvm":
        segmenter, rvm_device = create_rvm_model(rvm_model_path, device)
        rvm_state_left = [None] * 4
        rvm_state_right = [None] * 4
    else:
        segmenter = create_segmenter(model_path, device)
        rvm_state_left = None
        rvm_state_right = None
        rvm_device = None

    video = cv2.VideoCapture(input_video_path)
    if not video.isOpened():
        raise ValueError(f"Could not open video: {input_video_path}")

    fps, width, height = get_video_properties(video)
    effective_temporal_smoothing = temporal_smoothing
    if (
        0.0 < temporal_smoothing < 1.0
        and fps > 0
        and temporal_smoothing_reference_fps > 0
    ):
        effective_temporal_smoothing = temporal_smoothing ** (
            temporal_smoothing_reference_fps / fps
        )
    writer = RawVideoWriter(
        output_video_path,
        width,
        height,
        fps,
        codec="libx264",
        crf=crf,
        preset=preset,
        pixel_format="yuv420p",
    )

    previous_left_mask = None
    previous_right_mask = None
    frame_index = 0

    try:
        while True:
            success, frame_bgr = video.read()
            if not success:
                break

            frame_rgb = (
                cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
            )
            left_frame, right_frame = split_sbs_frame(frame_rgb)
            if source_fps > 0 and fps > 0:
                depth_index = min(
                    round(frame_index * source_fps / fps), len(depth_maps) - 1
                )
            else:
                depth_index = min(frame_index, len(depth_maps) - 1)
            depth_frame = resize_depth(
                depth_maps[depth_index], left_frame.shape[0], left_frame.shape[1]
            )

            if model_type == "rvm":
                left_mask, rvm_state_left = get_foreground_mask_rvm(
                    segmenter,
                    left_frame,
                    rvm_state_left,
                    rvm_device,
                    rvm_downsample_ratio,
                )
            else:
                left_mask = get_foreground_mask(
                    segmenter,
                    left_frame,
                    depth_frame,
                    foreground_class_set,
                    selection_mode,
                    threshold,
                    main_subject_score_ratio,
                    min_segment_area,
                    max_segment_area,
                    close_percentile,
                    min_depth_component_area,
                )

            if stereo_mask_mode == "left":
                right_mask = left_mask.copy()
            elif stereo_mask_mode == "depth":
                right_mask = project_mask_to_right_eye(
                    left_mask,
                    depth_frame,
                    max_disp,
                    max_disp_reference_width,
                    convergence,
                )
            else:
                if model_type == "rvm":
                    right_mask, rvm_state_right = get_foreground_mask_rvm(
                        segmenter,
                        right_frame,
                        rvm_state_right,
                        rvm_device,
                        rvm_downsample_ratio,
                    )
                else:
                    right_mask = get_foreground_mask(
                        segmenter,
                        right_frame,
                        depth_frame,
                        foreground_class_set,
                        selection_mode,
                        threshold,
                        main_subject_score_ratio,
                        min_segment_area,
                        max_segment_area,
                        close_percentile,
                        min_depth_component_area,
                    )
                if stereo_mask_mode == "union":
                    union_mask = np.maximum(left_mask, right_mask)
                    left_mask = union_mask
                    right_mask = union_mask.copy()

            left_mask = process_mask(
                left_mask,
                previous_left_mask,
                effective_temporal_smoothing,
                mask_erode,
                mask_dilate,
                mask_feather,
            )
            right_mask = process_mask(
                right_mask,
                previous_right_mask,
                effective_temporal_smoothing,
                mask_erode,
                mask_dilate,
                mask_feather,
            )

            left_output = composite_green(left_frame, left_mask, green_color)
            right_output = composite_green(right_frame, right_mask, green_color)
            output_rgb = np.concatenate([left_output, right_output], axis=1)
            writer.write(output_rgb)

            previous_left_mask = left_mask
            previous_right_mask = right_mask
            frame_index += 1
            if frame_index % 25 == 0:
                print(f"==> green-screened {frame_index} frames", flush=True)
    finally:
        video.release()
        writer.close()

    if write_metadata:
        print("==> writing chroma-key metadata", flush=True)
        green_metadata_color = np.clip(green_color * 255.0, 0, 255).astype(np.uint8)
        write_mp4_metadata(output_video_path, green_metadata_color, metadata_items)

    print(f"==> saved green-screen video: {output_video_path}", flush=True)


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


def parse_classes(foreground_classes):
    return {
        foreground_class.strip().lower()
        for foreground_class in foreground_classes.split(",")
        if foreground_class.strip()
    }


def load_depth_maps(depth_npz_path):
    if not os.path.isfile(depth_npz_path):
        raise FileNotFoundError(f"Depth map not found: {depth_npz_path}")

    depth_data = np.load(depth_npz_path)
    if "depth" not in depth_data:
        raise ValueError(
            f"Depth npz does not contain a 'depth' array: {depth_npz_path}"
        )

    return depth_data["depth"].astype(np.float32)


def resize_depth(depth, height, width):
    return cv2.resize(depth, (width, height), interpolation=cv2.INTER_LINEAR).astype(
        np.float32
    )


def load_convergence(depth_npz_path, fallback):
    meta_path = os.path.splitext(depth_npz_path)[0] + "_meta.json"
    if not os.path.isfile(meta_path):
        return fallback

    with open(meta_path) as meta_file:
        metadata = json.load(meta_file)
    return float(metadata.get("convergence", fallback))


def project_mask_to_right_eye(
    left_mask, depth_frame, max_disp, max_disp_reference_width, convergence
):
    height, width = left_mask.shape
    effective_max_disp = max_disp * width / max_disp_reference_width
    disp_pixels = (depth_frame - convergence) * 2.0 * effective_max_disp

    xs = np.arange(width, dtype=np.float32)
    ys = np.arange(height, dtype=np.float32)
    map_x, map_y = np.meshgrid(xs, ys)
    map_x = np.clip(map_x + disp_pixels, 0.0, width - 1).astype(np.float32)

    return cv2.remap(left_mask, map_x, map_y, cv2.INTER_LINEAR)


def create_rvm_model(rvm_model_path, device):
    if rvm_model_path in ("resnet50", "mobilenetv3"):
        model = torch.hub.load(
            "PeterL1n/RobustVideoMatting",
            rvm_model_path,
            trust_repo=True,
        )
    else:
        arch = "mobilenetv3" if "mobilenet" in rvm_model_path.lower() else "resnet50"
        model = torch.hub.load(
            "PeterL1n/RobustVideoMatting",
            arch,
            pretrained=False,
            trust_repo=True,
        )
        model.load_state_dict(torch.load(rvm_model_path, map_location="cpu"))

    rvm_device = torch.device(
        "cuda" if device == "cuda" and torch.cuda.is_available() else "cpu"
    )
    model = model.to(rvm_device).eval()
    return model, rvm_device


def get_foreground_mask_rvm(
    model, frame_rgb, recurrent_state, device, downsample_ratio
):
    frame_tensor = (
        torch.from_numpy(frame_rgb).permute(2, 0, 1).unsqueeze(0).float().to(device)
    )
    with torch.no_grad():
        _fgr, pha, *new_state = model(frame_tensor, *recurrent_state, downsample_ratio)
    return pha.squeeze().cpu().numpy(), new_state


def create_segmenter(model_path, device):
    from transformers import pipeline

    device_index = -1
    if device == "cuda" and torch.cuda.is_available():
        device_index = 0
    elif isinstance(device, int):
        device_index = device

    return pipeline("image-segmentation", model=model_path, device=device_index)


def get_segments(segmenter, frame_rgb, threshold):
    image = Image.fromarray((np.clip(frame_rgb, 0.0, 1.0) * 255.0).astype(np.uint8))
    raw_segments = segmenter(image)
    segments = []

    for segment in raw_segments:
        label = str(segment.get("label", "")).lower()
        score = float(segment.get("score", 1.0))
        if score < threshold:
            continue

        segment_mask = np.array(segment["mask"], dtype=np.float32)
        if segment_mask.max() > 1.0:
            segment_mask = segment_mask / 255.0
        segments.append(
            {
                "label": label,
                "score": score,
                "mask": np.clip(segment_mask, 0.0, 1.0),
            }
        )

    return segments


def get_depth_connected_mask(depth_frame, close_percentile, min_component_area):
    close_threshold = np.percentile(depth_frame, close_percentile)
    close_mask = (depth_frame >= close_threshold).astype(np.uint8)
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(close_mask, 8)
    connected_mask = np.zeros_like(close_mask, dtype=np.float32)

    height, width = depth_frame.shape
    min_area = max(1, int(height * width * min_component_area))

    for label_index in range(1, num_labels):
        area = stats[label_index, cv2.CC_STAT_AREA]
        if area >= min_area:
            connected_mask[labels == label_index] = 1.0

    return connected_mask


def get_segment_score(
    segment, depth_frame, depth_connected_mask, min_segment_area, max_segment_area
):
    mask = segment["mask"]
    hard_mask = mask > 0.5
    area = float(hard_mask.mean())
    if area < min_segment_area or area > max_segment_area:
        return None

    height, width = mask.shape
    y_indices, x_indices = np.nonzero(hard_mask)
    if len(x_indices) == 0:
        return None

    center_x = float(x_indices.mean()) / max(width - 1, 1)
    center_y = float(y_indices.mean()) / max(height - 1, 1)
    center_distance = np.sqrt((center_x - 0.5) ** 2 + (center_y - 0.55) ** 2)
    center_score = max(0.0, 1.0 - center_distance / 0.75)

    segment_depth = depth_frame[hard_mask]
    depth_score = float(np.percentile(segment_depth, 85))
    close_overlap = float(depth_connected_mask[hard_mask].mean())
    area_score = min(area / 0.2, 1.0)
    confidence_score = float(segment["score"])

    score = (
        depth_score * 0.45
        + close_overlap * 0.25
        + center_score * 0.15
        + area_score * 0.10
        + confidence_score * 0.05
    )

    return score


def select_main_subject_mask(
    segments,
    depth_frame,
    foreground_classes,
    selection_mode,
    main_subject_score_ratio,
    min_segment_area,
    max_segment_area,
    close_percentile,
    min_depth_component_area,
):
    mask = np.zeros(depth_frame.shape, dtype=np.float32)

    if selection_mode == "classes":
        for segment in segments:
            if foreground_classes and segment["label"] not in foreground_classes:
                continue
            mask = np.maximum(mask, segment["mask"])
        return mask

    if selection_mode == "all_segments":
        for segment in segments:
            area = float((segment["mask"] > 0.5).mean())
            if min_segment_area <= area <= max_segment_area:
                mask = np.maximum(mask, segment["mask"])
        return mask

    depth_connected_mask = get_depth_connected_mask(
        depth_frame, close_percentile, min_depth_component_area
    )
    scored_segments = []

    for segment in segments:
        score = get_segment_score(
            segment,
            depth_frame,
            depth_connected_mask,
            min_segment_area,
            max_segment_area,
        )
        if score is not None:
            scored_segments.append((score, segment))

    if not scored_segments:
        return depth_connected_mask

    scored_segments.sort(key=lambda item: item[0], reverse=True)
    best_score = scored_segments[0][0]
    keep_threshold = best_score * main_subject_score_ratio

    for score, segment in scored_segments:
        if score >= keep_threshold:
            mask = np.maximum(mask, segment["mask"])

    return mask


def get_foreground_mask(
    segmenter,
    frame_rgb,
    depth_frame,
    foreground_classes,
    selection_mode,
    threshold,
    main_subject_score_ratio,
    min_segment_area,
    max_segment_area,
    close_percentile,
    min_depth_component_area,
):
    segments = get_segments(segmenter, frame_rgb, threshold)
    return select_main_subject_mask(
        segments,
        depth_frame,
        foreground_classes,
        selection_mode,
        main_subject_score_ratio,
        min_segment_area,
        max_segment_area,
        close_percentile,
        min_depth_component_area,
    )


def process_mask(
    mask, previous_mask, temporal_smoothing, mask_erode, mask_dilate, mask_feather
):
    mask = np.clip(mask, 0.0, 1.0).astype(np.float32)

    if mask_erode > 0:
        kernel_size = mask_erode * 2 + 1
        kernel = np.ones((kernel_size, kernel_size), np.uint8)
        mask = cv2.erode(mask, kernel)

    if mask_dilate > 0:
        kernel_size = mask_dilate * 2 + 1
        kernel = np.ones((kernel_size, kernel_size), np.uint8)
        mask = cv2.dilate(mask, kernel)

    if mask_feather > 0:
        kernel_size = mask_feather * 2 + 1
        if kernel_size % 2 == 0:
            kernel_size += 1
        mask = cv2.GaussianBlur(mask, (kernel_size, kernel_size), 0)

    if previous_mask is not None and temporal_smoothing > 0.0:
        mask = previous_mask * temporal_smoothing + mask * (1.0 - temporal_smoothing)

    return np.clip(mask, 0.0, 1.0).astype(np.float32)


def composite_green(frame_rgb, mask, green_color):
    alpha = mask[:, :, None]

    spill_weight = np.clip(1.0 - alpha * 2.0, 0.0, 1.0)
    suppressed = frame_rgb.copy()
    suppressed[:, :, 1] = (
        frame_rgb[:, :, 1] * (1.0 - spill_weight[:, :, 0])
        + (frame_rgb[:, :, 0] + frame_rgb[:, :, 2]) * 0.5 * spill_weight[:, :, 0]
    )

    green_frame = np.ones_like(frame_rgb, dtype=np.float32) * green_color.reshape(
        1, 1, 3
    )
    return suppressed * alpha + green_frame * (1.0 - alpha)


def split_sbs_frame(frame_rgb):
    height, width = frame_rgb.shape[:2]
    if width % 2 != 0:
        raise ValueError(f"SBS video width must be even, got: {width}")

    half_width = width // 2
    return frame_rgb[:, :half_width], frame_rgb[:, half_width:]


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
    Fire(monitor_step("Step 5 - Greenscreen")(main))
