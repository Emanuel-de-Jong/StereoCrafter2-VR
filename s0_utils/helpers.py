import gc
import subprocess
from pathlib import Path

import imageio_ffmpeg
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


class RawVideoWriter:
    def __init__(
        self,
        output_path,
        width,
        height,
        fps,
        codec="libx264",
        crf=12,
        preset="slow",
        pixel_format="yuv420p",
    ):
        ffmpeg_path = imageio_ffmpeg.get_ffmpeg_exe()
        command = [
            ffmpeg_path,
            "-y",
            "-f",
            "rawvideo",
            "-vcodec",
            "rawvideo",
            "-pix_fmt",
            "rgb24",
            "-s",
            f"{width}x{height}",
            "-r",
            str(fps),
            "-i",
            "-",
            "-an",
            "-c:v",
            codec,
        ]

        if codec == "ffv1":
            command.extend(
                [
                    "-level",
                    "3",
                    "-coder",
                    "1",
                    "-context",
                    "1",
                    "-g",
                    "1",
                    "-pix_fmt",
                    "bgr0",
                ]
            )
        else:
            command.extend(
                [
                    "-crf",
                    str(crf),
                    "-preset",
                    preset,
                    "-pix_fmt",
                    pixel_format,
                ]
            )

        command.append(str(output_path))
        self.process = subprocess.Popen(command, stdin=subprocess.PIPE)

    def write(self, frame):
        if self.process.stdin is None:
            raise RuntimeError("Video writer is closed")

        frame = np.asarray(frame)
        if frame.dtype != np.uint8:
            if frame.max() <= 1.0:
                frame = frame * 255.0
            frame = np.clip(frame, 0.0, 255.0).astype(np.uint8)
        frame = np.ascontiguousarray(frame)
        self.process.stdin.write(frame.tobytes())

    def close(self):
        if self.process.stdin is not None:
            self.process.stdin.close()
        return_code = self.process.wait()
        if return_code != 0:
            raise subprocess.CalledProcessError(return_code, self.process.args)
