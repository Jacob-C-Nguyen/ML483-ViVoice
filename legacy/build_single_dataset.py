#!/usr/bin/env python3
"""
build_single_dataset.py

Creates a single dataset folder with all sentence-level wavs and a single metadata.csv
with pipe-separated columns: filename|text|language|speaker

Usage:
    python build_single_dataset.py
"""

import os
import json
import csv
import math
import soundfile as sf
import numpy as np

# ---- CONFIG ----
WHISPER_DIR = "whisper_out"   # input folder with whisperx outputs (one subfolder per audio)
OUTPUT_DIR = "dataset"        # output dataset folder
CLIPS_DIR = os.path.join(OUTPUT_DIR, "clips")
METADATA_CSV = os.path.join(OUTPUT_DIR, "metadata.csv")
LANG_CODE = "en"              # language code to put in CSV
SENTENCE_END_CHARS = {'.', '?', '!', '…'}

os.makedirs(CLIPS_DIR, exist_ok=True)

def collect_words_from_json(json_path):
    """Return flattened list of word dicts with keys: 'word','start','end'"""
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    words = []
    segments = data.get("segments", [])
    for seg in segments:
        if "words" in seg and isinstance(seg["words"], list):
            for w in seg["words"]:
                # Some whisperx outputs use 'word' or 'text' keys; prefer 'word'
                word_text = w.get("word") or w.get("text") or ""
                # Some tools include leading/trailing whitespace → normalize
                word_text = word_text.strip()
                start = float(w.get("start", 0.0))
                end = float(w.get("end", 0.0))
                if word_text == "":
                    continue
                words.append({"word": word_text, "start": start, "end": end})
    return words

def split_words_to_sentences(words):
    """Group words into sentences by punctuation at the end of words."""
    sentences = []
    curr = []
    for w in words:
        curr.append(w)
        # check last character of the word for sentence end punctuation
        last_char = w["word"][-1] if w["word"] else ""
        if last_char in SENTENCE_END_CHARS:
            sentences.append(curr)
            curr = []
    # if leftover words exist, save them as final sentence
    if curr:
        sentences.append(curr)
    return sentences

def build_text_from_words(word_list):
    """Reconstruct readable text: keep punctuation attached, avoid spaces before punctuation."""
    pieces = []
    for w in word_list:
        token = w["word"]
        if not token:
            continue
        # if token is punctuation-only (rare) append directly
        if all(ch in SENTENCE_END_CHARS or not ch.isalnum() for ch in token):
            if pieces:
                pieces[-1] = pieces[-1] + token
            else:
                pieces.append(token)
            continue

        # if token ends with punctuation (e.g. "world.") attach without extra space
        if token[-1] in SENTENCE_END_CHARS:
            pieces.append(token)
        else:
            # normal word: append with space
            pieces.append(token)
    # Now join with spaces but ensure punctuation tokens are not double spaced
    out = []
    for t in pieces:
        if not out:
            out.append(t)
        else:
            # if t starts with punctuation char, attach to previous
            if t[0] in SENTENCE_END_CHARS:
                out[-1] = out[-1] + t
            else:
                out.append(t)
    return " ".join(out).strip()

def export_clip_and_row(full_wav_path, sr, sentence_words, out_filename):
    """Export a WAV clip from full file using sample indices and return CSV row"""
    s = min(w["start"] for w in sentence_words)
    e = max(w["end"] for w in sentence_words)
    # convert to sample indices
    start_sample = max(0, int(math.floor(s * sr)))
    end_sample = max(start_sample + 1, int(math.ceil(e * sr)))
    # read region using soundfile (read entire file once outside if many sentences)
    data, file_sr = sf.read(full_wav_path, dtype="float32")
    if file_sr != sr:
        # unexpected sample rate: resample in-memory
        import numpy as _np
        _ratio = sr / file_sr
        # naive resample using numpy (simple) — better to pre-resample inputs beforehand
        data = _np.interp(
            _np.arange(0, int(len(data) * _ratio)),
            _np.arange(0, len(data)),
            data
        )
    clip = data[start_sample:end_sample]
    # write clip
    clip_path_abs = os.path.join(CLIPS_DIR, out_filename)
    sf.write(clip_path_abs, clip, sr)
    # build text
    text = build_text_from_words(sentence_words)
    return out_filename, text

def main():
    # scan whisper_out subfolders
    subdirs = sorted([d for d in os.listdir(WHISPER_DIR) if os.path.isdir(os.path.join(WHISPER_DIR, d))])
    if not subdirs:
        print("No subfolders found in", WHISPER_DIR)
        return

    csv_rows = []
    speaker_idx = 1

    for sub in subdirs:
        folder = os.path.join(WHISPER_DIR, sub)
        # assume WAV and JSON named as <sub>.wav and <sub>.json (common WhisperX layout)
        wav_path = None
        json_path = None
        # find wav and json in folder
        for f in os.listdir(folder):
            if f.lower().endswith(".wav") and wav_path is None:
                wav_path = os.path.join(folder, f)
            if f.lower().endswith(".json") and json_path is None:
                json_path = os.path.join(folder, f)
        if not wav_path or not json_path:
            print(f"Skipping {sub}: missing wav or json (found wav={wav_path}, json={json_path})")
            continue

        print(f"Processing {sub} -> speaker{speaker_idx}")

        words = collect_words_from_json(json_path)
        if not words:
            print("  no words found in json, skipping.")
            continue

        sentences = split_words_to_sentences(words)
        # get sample rate once
        info = sf.info(wav_path)
        sr = info.samplerate

        # For each sentence produce clip and CSV row
        sent_count = 0
        for i, sent_words in enumerate(sentences, start=1):
            fname = f"speaker{speaker_idx}_sentence_{i:04d}.wav"
            # Export clip and get text
            out_fname, text = export_clip_and_row(wav_path, sr, sent_words, fname)
            # CSV row: filename|text|language|speaker
            speaker_name = f"speaker{speaker_idx}"
            csv_rows.append([out_fname, text, LANG_CODE, speaker_name])
            sent_count += 1

        print(f"  exported {sent_count} sentences for speaker{speaker_idx}")
        speaker_idx += 1

    # write single metadata.csv with pipe delimiter
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    with open(METADATA_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f, delimiter="|", quoting=csv.QUOTE_MINIMAL)
        for row in csv_rows:
            writer.writerow(row)

    print("DONE. Clips in:", CLIPS_DIR)
    print("Metadata:", METADATA_CSV)
    print(f"Total clips: {len(csv_rows)}")

if __name__ == "__main__":
    main()
