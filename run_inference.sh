#!/bin/bash

INPUT_VIDEO_PATH="${1:-./inputs/vid.mp4}"
FILENAME=$(basename "$INPUT_VIDEO_PATH")
BASENAME="${FILENAME%.*}"
OUTPUT_DIR="./outputs"
mkdir -p "$OUTPUT_DIR"

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

printf "=== STEP 1 ===\n"
python -u s1_depth_splatting_inference.py \
	--pre_trained_path ./weights/stable-video-diffusion-img2vid-xt-1-1 \
	--unet_path ./weights/DepthCrafter \
	--input_video_path "$INPUT_VIDEO_PATH" \
	--output_video_path "$OUTPUT_DIR/${BASENAME}_1_splatting.mp4" \
	--target_fps 15 \
	--max_res 768 \
	--window_size 49 \
	--overlap 10 \
	--decode_chunk_size 4 \
	--cpu_offload model \
	--save_depth True

printf "\n\n=== STEP 2 ===\n"
python -u s2_inpainting_inference.py \
	--pre_trained_path ./weights/Wan2.1-VACE-14B-diffusers \
	--transformer_path ./weights/StereoCrafter2-FP8 \
	--input_video_path "$OUTPUT_DIR/${BASENAME}_1_splatting.mp4" \
	--save_dir "$OUTPUT_DIR" \
	--tile_num 2 \
	--frames_chunk 17 \
	--frames_overlap 2 \
	--transformer_dtype fp8 \
	--transformer_cpu_offload none \
	--vae_cpu_offload manual \
	--inpaint_scale 0.5 \
	--inference_steps 5

printf "\n\n=== STEP 3 ===\n"
python -u s3_greenscreen.py \
	--input_video_path "$OUTPUT_DIR/${BASENAME}_2_sbs.mp4" \
	--output_video_path "$OUTPUT_DIR/${BASENAME}_3_greenscreen.mp4" \
	--depth_npz_path "$OUTPUT_DIR/${BASENAME}_1_splatting.npz" \
	--enabled True

printf "\n\n=== STEP 4 ===\n"
python -u s4_interpolation.py \
	--input_video_path "$OUTPUT_DIR/${BASENAME}_3_greenscreen.mp4" \
	--output_video_path "$OUTPUT_DIR/${BASENAME}_4_interp.mp4" \
	--target_fps 45

printf "\n\n=== STEP 5 ===\n"
python -u s5_upscale.py \
	--input_video_path "$OUTPUT_DIR/${BASENAME}_4_interp.mp4" \
	--output_video_path "$OUTPUT_DIR/${BASENAME}_5_upscale.mp4"

printf "\n\n=== STEP 6 ===\n"
python -u s6_green_cleanup.py \
	--input_video_path "$OUTPUT_DIR/${BASENAME}_5_upscale.mp4" \
	--output_video_path "$OUTPUT_DIR/${BASENAME}_6_result.mp4" \
	--enabled True
