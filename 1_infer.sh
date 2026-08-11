#!/bin/bash

CONVERT_START_SECONDS=$(date +%s)
INPUT_VIDEO_PATH="${1:-./in/vid.mp4}"
FILENAME=$(basename "$INPUT_VIDEO_PATH")
BASENAME="${FILENAME%.*}"
OUTPUT_DIR="./out"
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
	--input_video_path "$INPUT_VIDEO_PATH" \
	--output_video_path "$OUTPUT_DIR/${BASENAME}_1_splatting.mkv" \
	--pre_trained_path ./weights/stable-video-diffusion-img2vid-xt-1-1 \
	--unet_path ./weights/DepthCrafter \
	--target_fps 30 \
	--max_disp 26 \
	--max_res 896 \
	--num_denoising_steps 6 \
	--window_size 56 \
	--overlap 16 \
	--decode_chunk_size 8 \
	--cpu_offload model \
	--depth_blur 2 \
	--convergence 0.5

printf "\n\n=== STEP 2: STEREO INPAINTING ===\n"
python -u s1_pipeline/s2_inpainting.py \
	--input_video_path "$OUTPUT_DIR/${BASENAME}_1_splatting.mkv" \
	--source_video_path "$INPUT_VIDEO_PATH" \
	--output_video_path "$OUTPUT_DIR/${BASENAME}_2_sbs.mkv" \
	--anaglyph_video_path "$OUTPUT_DIR/${BASENAME}_2_anaglyph.mp4" \
	--output_path "$OUTPUT_DIR" \
	--pre_trained_path ./weights/Wan2.1-VACE-14B-diffusers \
	--transformer_path ./weights/StereoCrafter2-FP8 \
	--tile_num 2 \
	--frames_chunk 25 \
	--frames_overlap 6 \
	--transformer_dtype fp8 \
	--transformer_cpu_offload none \
	--vae_cpu_offload manual \
	--inpaint_scale 1.0 \
	--inference_steps 7

printf "\n\n=== STEP 3: GREENSCREEN ===\n"
python -u s1_pipeline/s3_greenscreen.py \
	--input_video_path "$OUTPUT_DIR/${BASENAME}_2_sbs.mkv" \
	--output_video_path "$OUTPUT_DIR/${BASENAME}_3_greenscreen.mkv" \
	--depth_npz_path "$OUTPUT_DIR/${BASENAME}_1_splatting.npz" \
	--enabled True

printf "\n\n=== STEP 4: INTERPOLATION ===\n"
python -u s1_pipeline/s4_interpolation.py \
	--input_video_path "$OUTPUT_DIR/${BASENAME}_3_greenscreen.mkv" \
	--output_video_path "$OUTPUT_DIR/${BASENAME}_4_interp.mp4" \
	--target_fps 60

printf "\n\n=== STEP 5: UPSCALE ===\n"
python -u s1_pipeline/s5_upscale.py \
	--input_video_path "$OUTPUT_DIR/${BASENAME}_4_interp.mp4" \
	--output_video_path "$OUTPUT_DIR/${BASENAME}_5_upscale.mp4" \
	--realesrgan_model realesr-animevideov3

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
