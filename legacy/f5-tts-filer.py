import os
import json
import shutil
import pandas as pd

# ==== CONFIG ====
CSV_FILE = "/home/jacob/Documents/GitHub/ML483-ViVoice/baseModelMaker/dataset/metadata_coqui_final.csv"
DATASET_ROOT = "f5_dataset"   # output folder you want to generate
AUDIO_ROOT = "/home/jacob/Documents/GitHub/ML483-ViVoice/baseModelMaker/dataset"  # folder where your mono/*.wav are stored
# =================

# Create dataset root
os.makedirs(DATASET_ROOT, exist_ok=True)

print("Loading CSV...")
df = pd.read_csv(CSV_FILE, delimiter="|", dtype=str)

print("Found rows:", len(df))

# F5-TTS metadata JSONL file
jsonl_path = os.path.join(DATASET_ROOT, "metadata.jsonl")
jsonl_file = open(jsonl_path, "w", encoding="utf-8")

for idx, row in df.iterrows():
    audio_path = row["audio_file"]              # example: mono/speaker1_sentence_0001.wav
    text = row["text"]
    speaker = row["speaker_name"]

    abs_audio_path = os.path.join(AUDIO_ROOT, audio_path)

    # Split file components
    filename = os.path.basename(audio_path)     # speaker1_sentence_0001.wav
    base = os.path.splitext(filename)[0]        # speaker1_sentence_0001

    # Speaker directory structure
    speaker_dir = os.path.join(DATASET_ROOT, speaker)
    wav_dir = os.path.join(speaker_dir, "wavs")
    txt_dir = os.path.join(speaker_dir, "txt")

    os.makedirs(wav_dir, exist_ok=True)
    os.makedirs(txt_dir, exist_ok=True)

    # Copy the WAV file into the speaker's wav directory
    target_wav_path = os.path.join(wav_dir, filename)
    shutil.copy(abs_audio_path, target_wav_path)

    # Create transcription .txt file
    target_txt_path = os.path.join(txt_dir, base + ".txt")
    with open(target_txt_path, "w", encoding="utf-8") as f:
        f.write(text.strip())

    # Write entry to metadata.jsonl
    entry = {
        "audio_filepath": target_wav_path,
        "text": text,
        "speaker": speaker
    }
    jsonl_file.write(json.dumps(entry) + "\n")

jsonl_file.close()

print("====================================")
print(" F5-TTS dataset generated successfully!")
print(" Output folder:", DATASET_ROOT)
print(" Total items written:", len(df))
print("====================================")
