import os
import json
import csv
import soundfile as sf

INPUT_DIR = "whisper_out"      # folder containing whisperx outputs
OUTPUT_CLIPS = "clips"         # base folder for wav clips
os.makedirs(OUTPUT_CLIPS, exist_ok=True)

def words_to_sentences(words):
    sentences = []
    current = []
    start_time = None
    end_time = None

    for w in words:
        if start_time is None:
            start_time = w["start"]

        current.append(w["word"])
        end_time = w["end"]

        if w["word"].strip().endswith(('.', '!', '?')):
            text = "".join(current).strip()
            sentences.append((start_time, end_time, text))
            current = []
            start_time = None

    if current:
        text = "".join(current).strip()
        sentences.append((start_time, end_time, text))

    return sentences


for subdir in os.listdir(INPUT_DIR):
    subpath = os.path.join(INPUT_DIR, subdir)
    if not os.path.isdir(subpath):
        continue

    # find corresponding files
    json_file = None
    wav_file = None

    for f in os.listdir(subpath):
        if f.endswith(".json"):
            json_file = os.path.join(subpath, f)
        if f.endswith(".wav"):
            wav_file = os.path.join(subpath, f)

    if not json_file or not wav_file:
        continue

    print(f"Processing speaker/video: {subdir}")

    # load audio
    audio, sr = sf.read(wav_file)

    # load whisper json
    with open(json_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    # flatten words
    words = []
    for seg in data["segments"]:
        if "words" in seg:
            words.extend(seg["words"])

    sentences = words_to_sentences(words)

    # speaker name from folder name
    speaker_name = subdir.replace(" ", "_")

    # folder for this speaker
    speaker_clip_dir = os.path.join(OUTPUT_CLIPS, speaker_name)
    os.makedirs(speaker_clip_dir, exist_ok=True)

    # CSV for this speaker
    csv_path = f"{speaker_name}.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["audio_path", "text", "speaker"])

        # process each sentence
        for i, (s, e, text) in enumerate(sentences):
            start_sample = int(s * sr)
            end_sample = int(e * sr)

            clip_audio = audio[start_sample:end_sample]
            out_name = f"{speaker_name}_sent_{i:04d}.wav"
            out_path = os.path.join(speaker_clip_dir, out_name)

            sf.write(out_path, clip_audio, sr)

            writer.writerow([out_path, text, speaker_name])

    print(f"✓ Created clips and CSV for {speaker_name}")

print("All speakers processed successfully.")
