import torch
from TTS.api import TTS
import os

# ----------------------------
# AUTO-SELECT CPU OR GPU
# ----------------------------
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# ----------------------------
# LOAD XTTS-V2 BASE MODEL
# ----------------------------
tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2").to(device)

# ----------------------------
# PATH TO TRAINING DATA
# ----------------------------
DATA_DIR = "./data"

# XTTS expects:
# data/speaker_name/audio.wav
# data/speaker_name/transcript.txt
# ----------------------------

# ----------------------------
# START FINE-TUNING
# ----------------------------
tts.finetune(
    output_path="./xtts_finetuned_output",
    data_path=DATA_DIR,
    epochs=50,                 # adjust based on dataset size
    batch_size=4,              # small batches for CPU or small GPU
    lr=0.0001,
    eval_split_size=0.1,       # 10% eval
    use_phonemes=True,         # recommended for XTTS
    max_audio_length=15,       # avoids extremely long clips
    device=device
)

print("Fine-tuning complete! Model saved in ./xtts_finetuned_output")
