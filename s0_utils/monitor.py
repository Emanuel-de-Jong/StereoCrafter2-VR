import inspect
import threading
import time
from functools import wraps

import psutil

import s0_utils.global_params as g
from s0_utils.helpers import format_duration


class StepMonitor:
    def __init__(
        self, step_name, sample_interval_seconds=g.RESOURCE_SAMPLE_INTERVAL_SECONDS
    ):
        self.step_name = step_name
        self.sample_interval_seconds = sample_interval_seconds
        self.samples = []
        self.start_time = None
        self.end_time = None
        self.stop_event = threading.Event()
        self.thread = None
        self.nvml = None
        self.gpu_handle = None
        self.start_disk_io = None
        self.previous_disk_io = None
        self.previous_sample_time = None

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.stop()
        self.print_summary()

    def start(self):
        self.start_time = time.time()
        self.start_disk_io = psutil.disk_io_counters()
        self.previous_disk_io = self.start_disk_io
        self.previous_sample_time = self.start_time
        self.setup_nvml()
        self.sample()
        self.thread = threading.Thread(target=self.sample_loop, daemon=True)
        self.thread.start()

    def stop(self):
        self.end_time = time.time()
        self.stop_event.set()
        if self.thread is not None:
            self.thread.join(timeout=1)
        self.sample()
        self.shutdown_nvml()

    def setup_nvml(self):
        try:
            import pynvml

            pynvml.nvmlInit()
            self.nvml = pynvml
            self.gpu_handle = pynvml.nvmlDeviceGetHandleByIndex(0)
        except Exception:
            self.nvml = None
            self.gpu_handle = None

    def shutdown_nvml(self):
        if self.nvml is None:
            return

        try:
            self.nvml.nvmlShutdown()
        except Exception:
            pass

    def sample_loop(self):
        while not self.stop_event.wait(self.sample_interval_seconds):
            self.sample()

    def sample(self):
        sample_time = time.time()
        disk_io = psutil.disk_io_counters()
        elapsed_seconds = max(sample_time - self.previous_sample_time, 0.001)

        gpu_vram_mb = None
        gpu_load_percent = None
        if self.nvml is not None and self.gpu_handle is not None:
            try:
                memory_info = self.nvml.nvmlDeviceGetMemoryInfo(self.gpu_handle)
                utilization = self.nvml.nvmlDeviceGetUtilizationRates(self.gpu_handle)
                gpu_vram_mb = memory_info.used / 1024 / 1024
                gpu_load_percent = float(utilization.gpu)
            except Exception:
                gpu_vram_mb = None
                gpu_load_percent = None

        self.samples.append(
            {
                "ram_gb": psutil.virtual_memory().used / 1024 / 1024 / 1024,
                "cpu_percent": psutil.cpu_percent(interval=None),
                "gpu_vram_mb": gpu_vram_mb,
                "gpu_load_percent": gpu_load_percent,
                "disk_read_mb_per_second": (
                    disk_io.read_bytes - self.previous_disk_io.read_bytes
                )
                / 1024
                / 1024
                / elapsed_seconds,
                "disk_write_mb_per_second": (
                    disk_io.write_bytes - self.previous_disk_io.write_bytes
                )
                / 1024
                / 1024
                / elapsed_seconds,
                "disk_read_total_mb": (
                    disk_io.read_bytes - self.start_disk_io.read_bytes
                )
                / 1024
                / 1024,
                "disk_write_total_mb": (
                    disk_io.write_bytes - self.start_disk_io.write_bytes
                )
                / 1024
                / 1024,
            }
        )
        self.previous_disk_io = disk_io
        self.previous_sample_time = sample_time

    def print_summary(self):
        duration_seconds = (self.end_time or time.time()) - self.start_time
        print(f"\n=== {self.step_name.upper()} SUMMARY ===", flush=True)
        print(f"Duration: {format_duration(duration_seconds)}", flush=True)
        self.print_average_max("VRAM used", "gpu_vram_mb", "MB")
        self.print_average_max("RAM used", "ram_gb", "GB")
        self.print_average_max("GPU load", "gpu_load_percent", "%")
        self.print_average_max("CPU load", "cpu_percent", "%")
        self.print_average_max("Disk read", "disk_read_mb_per_second", "MB/s")
        self.print_average_max("Disk write", "disk_write_mb_per_second", "MB/s")
        if self.samples:
            last_sample = self.samples[-1]
            print(
                f"Disk read total: {last_sample['disk_read_total_mb'] / 1024:.2f} GB",
                flush=True,
            )
            print(
                f"Disk write total: {last_sample['disk_write_total_mb'] / 1024:.2f} GB",
                flush=True,
            )

    def print_average_max(self, label, key, unit):
        values = [sample[key] for sample in self.samples if sample[key] is not None]
        if not values:
            print(f"{label}: N/A", flush=True)
            return

        average_value = sum(values) / len(values)
        max_value = max(values)
        print(
            f"{label}: avg {average_value:.2f} {unit}, max {max_value:.2f} {unit}",
            flush=True,
        )


def monitor_step(step_name):
    def decorator(function):
        @wraps(function)
        def wrapper(*args, **kwargs):
            with StepMonitor(step_name):
                return function(*args, **kwargs)

        wrapper.__signature__ = inspect.signature(function)
        return wrapper

    return decorator
