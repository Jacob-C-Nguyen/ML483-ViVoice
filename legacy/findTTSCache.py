import os
from pathlib import Path

# Coqui TTS cache locations
TTS_CACHE_DIRS = [
    Path.home() / ".local/share/tts",
    Path.home() / ".cache/tts",
    Path("/root/.local/share/tts"),
    Path("/root/.cache/tts"),
]

print("=== SEARCHING FOR TTS MODEL DIRECTORIES ===")
for d in TTS_CACHE_DIRS:
    print(f"\n📂 Checking: {d}")
    if not d.exists():
        print("   (does not exist)")
        continue
    for item in d.iterdir():
        if item.is_dir():
            print("   DIR:", item.name)
            # show config or model files if they exist
            cfg = item / "config.json"
            pth = item / "model.pth"
            if cfg.exists():
                print("      - config.json found")
            if pth.exists():
                print("      - model.pth found")
