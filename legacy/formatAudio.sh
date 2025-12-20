#!/bin/bash

INPUT_DIR="out24hz"
OUTPUT_DIR="whisper_out"

mkdir -p "$OUTPUT_DIR"

for f in "$INPUT_DIR"/*.wav; do
    
    if [ ! -e "$f" ]; then
        echo "No WAV files found in $INPUT_DIR"
        exit 1
    fi

    filename=$(basename "$f")

    # sanitize filename → remove illegal characters
    name=$(echo "$filename" | sed 's/[^A-Za-z0-9._-]/_/g')

    echo "Processing: $filename"
    echo "Saving to: $OUTPUT_DIR/$name"

    mkdir -p "$OUTPUT_DIR/$name"

    whisperx "$f" \
        --language en \
        --diarize \
        --hf_token "$HF_TOKEN" \
        --output_dir "$OUTPUT_DIR/$name"
    
    echo "Done: $filename"
    echo "----------------------------------------"
done
