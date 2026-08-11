#!/bin/bash

CONVERT_START_SECONDS=$(date +%s)
INPUT_VIDEO_PATH="${1:-./inputs/vid.mp4}"
FILENAME=$(basename "$INPUT_VIDEO_PATH")
BASENAME="${FILENAME%.*}"
OUTPUT_DIR="./outputs"
mkdir -p "$OUTPUT_DIR"

format_duration() {
	local duration_seconds="$1"
	local hours=$((duration_seconds / 3600))
	local minutes=$(((duration_seconds % 3600) / 60))
	local seconds=$((duration_seconds % 60))
	printf "%02d:%02d:%02d" "$hours" "$minutes" "$seconds"
}

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

printf "=== STEP 1: DEPTH SPLATTING ===\n"
python -u s1_pipeline/s1_depth_splatting.py \
	--pre_trained_path ./weights/stable-video-diffusion-img2vid-xt-1-1 \
	--unet_path ./weights/DepthCrafter \
	--input_video_path "$INPUT_VIDEO_PATH" \
	--output_video_path "$OUTPUT_DIR/${BASENAME}_1_splatting.mp4" \
	--max_disp 26 \
	--target_fps 15 \
	--max_res 768 \
	--window_size 49 \
	--overlap 10 \
	--decode_chunk_size 4 \
	--cpu_offload model

printf "\n\n=== STEP 2: STEREO INPAINTING ===\n"
python -u s1_pipeline/s2_inpainting.py \
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

printf "\n\n=== STEP 3: GREENSCREEN ===\n"
python -u s1_pipeline/s3_greenscreen.py \
	--input_video_path "$OUTPUT_DIR/${BASENAME}_2_sbs.mp4" \
	--output_video_path "$OUTPUT_DIR/${BASENAME}_3_greenscreen.mp4" \
	--depth_npz_path "$OUTPUT_DIR/${BASENAME}_1_splatting.npz" \
	--enabled True

printf "\n\n=== STEP 4: INTERPOLATION ===\n"
python -u s1_pipeline/s4_interpolation.py \
	--input_video_path "$OUTPUT_DIR/${BASENAME}_3_greenscreen.mp4" \
	--output_video_path "$OUTPUT_DIR/${BASENAME}_4_interp.mp4" \
	--target_fps 45

printf "\n\n=== STEP 5: UPSCALE ===\n"
python -u s1_pipeline/s5_upscale.py \
	--input_video_path "$OUTPUT_DIR/${BASENAME}_4_interp.mp4" \
	--output_video_path "$OUTPUT_DIR/${BASENAME}_5_upscale.mp4"

printf "\n\n=== STEP 6: GREEN CLEANUP ===\n"
python -u s1_pipeline/s6_green_cleanup.py \
	--input_video_path "$OUTPUT_DIR/${BASENAME}_5_upscale.mp4" \
	--output_video_path "$OUTPUT_DIR/${BASENAME}_6_result.mp4" \
	--enabled True

CONVERT_END_SECONDS=$(date +%s)
CONVERT_DURATION_SECONDS=$((CONVERT_END_SECONDS - CONVERT_START_SECONDS))

printf "\n\n=== FULL CONVERT SUMMARY ===\n"
printf "Input: %s\n" "$INPUT_VIDEO_PATH"
printf "Output: %s\n" "$OUTPUT_DIR/${BASENAME}_6_result.mp4"
printf "Duration: %s\n" "$(format_duration "$CONVERT_DURATION_SECONDS")"
