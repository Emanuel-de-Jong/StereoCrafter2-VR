#!/bin/bash

INPUT_DIR="./inputs"
OUTPUT_DIR="./outputs"
mkdir -p "$OUTPUT_DIR"

shopt -s nullglob

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

for video in "$INPUT_DIR"/*.{mp4,mov,avi,mkv,webm}; do
	# Check if a file actually exists to avoid running on literal '*.mp4' if empty
	[ -e "$video" ] || continue

	filename=$(basename "$video")
	basename="${filename%.*}"
	output_file="$OUTPUT_DIR/${basename}_1_splatting.mp4"

	printf "=== STEP 1 ==="
	python -u s1_depth_splatting_inference.py \
		--pre_trained_path ./weights/stable-video-diffusion-img2vid-xt-1-1 \
		--unet_path ./weights/DepthCrafter \
		--input_video_path "$video" \
		--output_video_path "$output_file" \
		--target_fps 15 \
		--max_res 768 \
		--window_size 49 \
		--overlap 10 \
		--decode_chunk_size 4 \
		--cpu_offload model

	printf "\n=== STEP 2 ==="
	python -u s2_inpainting_inference.py \
		--pre_trained_path ./weights/Wan2.1-VACE-14B-diffusers \
		--transformer_path ./weights/StereoCrafter2-FP8 \
		--input_video_path "$output_file" \
		--save_dir "$OUTPUT_DIR" \
		--tile_num 2 \
		--frames_chunk 17 \
		--frames_overlap 2 \
		--transformer_dtype fp8 \
		--transformer_cpu_offload none \
		--vae_cpu_offload manual \
		--inpaint_scale 0.5 \
		--inference_steps 6

	printf "\n=== STEP 3 ==="
	python -u s3_interpolation.py \
		--input_video_path "$OUTPUT_DIR/${basename}_2_sbs.mp4" \
		--output_video_path "$OUTPUT_DIR/${basename}_3_interp.mp4" \
		--target_fps 45

	printf "\n=== STEP 4 ==="
	python -u s4_upscale.py \
		--input_video_path "$OUTPUT_DIR/${basename}_3_interp.mp4" \
		--output_video_path "$OUTPUT_DIR/${basename}_4_upscale.mp4"

	printf "\n-----------------------------------\n"
done
