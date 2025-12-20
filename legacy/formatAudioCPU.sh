#!/bin/bash

INPUT_DIR="out24hz"
OUTPUT_DIR="whisper_out"
HF_TOKEN="token here"

mkdir -p "$OUTPUT_DIR"

for f in "$INPUT_DIR"/*.wav; do
    name=$(basename "$f" .wav)

    echo "Processing: $name"

    whisperx "$f" \
        --language en \
        --compute_type float32 \
        --device cpu \
        --hf_token "$HF_TOKEN" \
        --output_dir "$OUTPUT_DIR/$name" \
        --output_format json

    echo "Done: $name"
done
