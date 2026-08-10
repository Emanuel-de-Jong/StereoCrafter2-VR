#!/bin/bash

INPUT_DIR="./inputs"
OUTPUT_DIR="./outputs"

shopt -s nullglob

for video in "$INPUT_DIR"/*.{mp4,mov,avi,mkv,webm}; do
    # Check if a file actually exists to avoid running on literal '*.mp4' if empty
    [ -e "$video" ] || continue

	filename=$(basename "$video")
    basename="${filename%.*}"
	output_file="$OUTPUT_DIR/${basename}_splatting_results.mp4"

    echo "Processing $video..."

	python depth_splatting_inference.py \
		--pre_trained_path ./weights/stable-video-diffusion-img2vid-xt-1-1 \
		--unet_path ./weights/DepthCrafter \
		--input_video_path "$video" \
		--output_video_path "$output_file"

	python inpainting_inference.py \
		--pre_trained_path ./weights/Wan2.1-VACE-14B-diffusers \
		--unet_path ./weights/StereoCrafter2 \
		--input_video_path "$output_file" \
        --save_dir "$OUTPUT_DIR" \
		--tile_num 2

    echo "Finished $video"
    echo "-----------------------------------"
done
