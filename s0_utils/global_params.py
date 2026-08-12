from pathlib import Path

BASE_PATH = Path(__file__).resolve().parent.parent

INPUTS_DIR = BASE_PATH / "in"
OUTPUTS_DIR = BASE_PATH / "out"
WEIGHTS_DIR = BASE_PATH / "weights"
DEPENDENCIES_DIR = BASE_PATH / "dependencies"

SVD_WEIGHTS_PATH = WEIGHTS_DIR / "stable-video-diffusion-img2vid-xt-1-1"
DEPTHCRAFTER_WEIGHTS_PATH = WEIGHTS_DIR / "DepthCrafter"
WAN_WEIGHTS_PATH = WEIGHTS_DIR / "Wan2.1-VACE-14B-diffusers"
STEREOCRAFTER_WEIGHTS_PATH = WEIGHTS_DIR / "StereoCrafter2-FP8"
VIDEO2X_PATH = DEPENDENCIES_DIR / "Video2X" / "Video2X-x86_64.AppImage"

RESOURCE_SAMPLE_INTERVAL_SECONDS = 10
