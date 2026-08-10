#!/bin/bash

INPUT_DIR="./inputs"
OUTPUT_DIR="./outputs"
mkdir -p "$OUTPUT_DIR"

python depth_splatting_inference.py \
	--pre_trained_path ./weights/stable-video-diffusion-img2vid-xt-1-1 \
	--unet_path ./weights/DepthCrafter \
	--input_video_path "$INPUT_DIR/vid.mp4" \
	--output_video_path "$OUTPUT_DIR/vid_splatting_results.mp4"

python inpainting_inference.py \
    --pre_trained_path ./weights/Wan2.1-VACE-14B-diffusers \
    --transformer_path ./weights/StereoCrafter2 \
    --input_video_path "$OUTPUT_DIR/vid_splatting_results.mp4" \
    --save_dir "$OUTPUT_DIR" \
    --tile_num 2
