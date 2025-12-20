import json, subprocess, os, sys

audio = sys.argv[1]
json_file = sys.argv[2]
output_dir = "sentence_clips"
os.makedirs(output_dir, exist_ok=True)

data = json.load(open(json_file))

for i, seg in enumerate(data["segments"]):
    start = seg["start"]
    end = seg["end"]
    out = f"{output_dir}/sentence_{i+1}.wav"

    subprocess.run([
        "ffmpeg", "-y",
        "-i", audio,
        "-ss", str(start),
        "-to", str(end),
        out
    ])
