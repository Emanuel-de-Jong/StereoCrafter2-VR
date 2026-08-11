import os
import math
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path

import imageio_ffmpeg
from fire import Fire

sys.path.append(str(Path(__file__).resolve().parent.parent))

import s0_utils.global_params as g
from s0_utils.helpers import get_video_fps, run_command, should_skip_output
from s0_utils.monitor import monitor_step
from s1_pipeline.step_contracts import StepResult


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


@dataclass
class InterpolationConfig:
    input_video_path: str = str(g.OUTPUTS_DIR / "vid_3_greenscreen.mp4")
    output_video_path: str = str(g.OUTPUTS_DIR / "vid_4_interp.mp4")
    video2x_path: str = str(g.VIDEO2X_PATH)
    target_fps: int = g.INTERPOLATION_TARGET_FPS
    rife_model: str = "rife-v4.25"
    gpu: int = 0
    scene_thresh: int = 100
    crf: int = 16
    preset: str = "medium"
    overwrite: bool = False


def run(config: InterpolationConfig) -> StepResult:
    if should_skip_output(config.output_video_path, config.overwrite):
        return StepResult(config.output_video_path, skipped=True)

    if not os.path.isfile(config.input_video_path):
        raise FileNotFoundError(f"Input video not found: {config.input_video_path}")
    if not os.path.isfile(config.video2x_path):
        raise FileNotFoundError(f"Video2X AppImage not found: {config.video2x_path}")
    if config.target_fps <= 0:
        raise ValueError(f"target_fps must be greater than 0, got: {config.target_fps}")

    os.makedirs(os.path.dirname(config.output_video_path) or ".", exist_ok=True)

    input_fps = get_video_fps(config.input_video_path)
    frame_rate_multiplier = get_frame_rate_multiplier(input_fps, config.target_fps)
    interpolated_fps = input_fps * frame_rate_multiplier
    print(f"Input FPS: {input_fps:.3f}", flush=True)
    print(f"Target FPS: {config.target_fps}", flush=True)

    if input_fps >= config.target_fps:
        print("Input FPS is already at or above target FPS, copying video.", flush=True)
        shutil.copy2(config.input_video_path, config.output_video_path)
        return StepResult(config.output_video_path)

    print(f"RIFE model: {config.rife_model}", flush=True)
    print(f"Frame rate multiplier: {frame_rate_multiplier}x", flush=True)

    if abs(interpolated_fps - config.target_fps) < 0.01:
        run_rife_interpolation(
            config.video2x_path,
            config.input_video_path,
            config.output_video_path,
            frame_rate_multiplier,
            config.rife_model,
            config.gpu,
            config.scene_thresh,
        )
        return StepResult(config.output_video_path)

    temp_output_path = os.path.join(
        os.path.dirname(config.output_video_path) or ".",
        f".{os.path.splitext(os.path.basename(config.output_video_path))[0]}_rife.mp4",
    )

    try:
        run_rife_interpolation(
            config.video2x_path,
            config.input_video_path,
            temp_output_path,
            frame_rate_multiplier,
            config.rife_model,
            config.gpu,
            config.scene_thresh,
        )
        conform_video_fps(
            temp_output_path,
            config.output_video_path,
            config.target_fps,
            config.crf,
            config.preset,
        )
    finally:
        if os.path.exists(temp_output_path):
            os.remove(temp_output_path)

    return StepResult(config.output_video_path)


def main(
    input_video_path=str(g.OUTPUTS_DIR / "vid_3_greenscreen.mp4"),
    output_video_path=str(g.OUTPUTS_DIR / "vid_4_interp.mp4"),
    video2x_path=str(g.VIDEO2X_PATH),
    target_fps=g.INTERPOLATION_TARGET_FPS,
    rife_model="rife-v4.25",
    gpu=0,
    scene_thresh=100,
    crf=16,
    preset="medium",
    overwrite=False,
):
    config = InterpolationConfig(
        input_video_path=input_video_path,
        output_video_path=output_video_path,
        video2x_path=video2x_path,
        target_fps=target_fps,
        rife_model=rife_model,
        gpu=gpu,
        scene_thresh=scene_thresh,
        crf=crf,
        preset=preset,
        overwrite=overwrite,
    )
    return run(config)


if __name__ == "__main__":
    Fire(monitor_step("Step 4 - Interpolation")(main))
