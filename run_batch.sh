#!/bin/bash

INPUT_DIR="./inputs"

shopt -s nullglob

for video in "$INPUT_DIR"/*.{mp4,mov,avi,mkv,webm}; do
	# Check if a file actually exists to avoid running on literal '*.mp4' if empty
	[ -e "$video" ] || continue

	printf "=== BATCH INPUT: %s ===\n" "$video"
	./run_inference.sh "$video"
	printf "\n\n-----------------------------------\n\n"
done
