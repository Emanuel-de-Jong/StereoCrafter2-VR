import gc
import subprocess
from pathlib import Path

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
