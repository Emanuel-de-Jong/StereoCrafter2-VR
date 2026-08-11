#!/bin/bash

INPUT_DIR="./in"

shopt -s nullglob

for video in "$INPUT_DIR"/*.{mp4,mov,avi,mkv,webm}; do
	# Check if a file actually exists to avoid running on literal '*.mp4' if empty
	[ -e "$video" ] || continue

	printf "=== BATCH INPUT: %s ===\n" "$video"
	./1_infer.sh "$video"
	printf "\n\n-----------------------------------\n\n"
done
