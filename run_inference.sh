#!/bin/bash

INPUT_DIR="./inputs"
OUTPUT_DIR="./outputs"
mkdir -p "$OUTPUT_DIR"

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

python -u depth_splatting_inference.py \
	--pre_trained_path ./weights/stable-video-diffusion-img2vid-xt-1-1 \
	--unet_path ./weights/DepthCrafter \
	--input_video_path "$INPUT_DIR/vid.mp4" \
	--output_video_path "$OUTPUT_DIR/vid_splatting_results.mp4" \
	--target_fps 15 \
	--max_res 1024 \
	--window_size 70 \
	--overlap 25 \
	--decode_chunk_size 8 \
	--cpu_offload model

# python -u inpainting_inference.py \
#     --pre_trained_path ./weights/Wan2.1-VACE-14B-diffusers \
#     --transformer_path ./weights/StereoCrafter2 \
#     --input_video_path "$OUTPUT_DIR/vid_splatting_results.mp4" \
#     --save_dir "$OUTPUT_DIR" \
#     --tile_num 2
