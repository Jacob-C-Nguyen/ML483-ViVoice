#!/bin/bash

# Activate your conda environment
source ~/anaconda3/etc/profile.d/conda.sh
conda activate xtts2

# Navigate to your project folder
cd /home/jacob/Documents/GitHub/ML483-ViVoice/baseModelMaker

# Run XTTS training
python -m trainer.distribute --script stage1.py

tts-train --config_path TTS/recipes/ljspeech/vits_tts/config.json \
          --model vits \
          --coqpit.datasets.0.path /absolute/path/to/processed_voice_data \
          --coqpit.datasets.0.meta_file_train metadata.csv \
          --coqpit.output_path ./training_output