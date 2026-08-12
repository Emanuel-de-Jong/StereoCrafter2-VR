#!/bin/bash

INPUT_DIR="./in"
OUTPUT_DIR="./out"

shopt -s nullglob

videos=()

for video in "$INPUT_DIR"/*.{mp4,mov,avi,mkv,webm}; do
	[ -e "$video" ] || continue

	filename="$(basename "$video")"
	basename="${filename%.*}"
	result="$OUTPUT_DIR/${basename}_5_result.mp4"

	if [ -e "$result" ]; then
		continue
	fi

	videos+=("$video")
done

TOTAL_COUNT=${#videos[@]}
CURRENT_COUNT=0

for video in "${videos[@]}"; do
	((CURRENT_COUNT++))

	printf "=== %s (%d/%d) ===\n" "$video" "$CURRENT_COUNT" "$TOTAL_COUNT"

	./1_infer.sh "$video"

	printf "\n\n-----------------------------------\n\n"
done
