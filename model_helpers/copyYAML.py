#!/usr/bin/env python3
"""
Copy F5TTS_Fine_Tuned.yaml into the installed f5_tts config directory
"""

import shutil
import sys
from pathlib import Path
import site


def find_f5tts_config_dir():
    """
    Locate f5_tts/configs inside site-packages
    """
    for sp in site.getsitepackages():
        candidate = Path(sp) / "f5_tts" / "configs"
        if candidate.exists():
            return candidate
    return None


def main():
    src = Path("F5TTS_Fine_Tuned.yaml")

    if not src.exists():
        print(f"❌ Source file not found: {src}")
        sys.exit(1)

    dest_dir = find_f5tts_config_dir()

    if dest_dir is None:
        print("❌ Could not find f5_tts/configs directory")
        sys.exit(1)

    dest = dest_dir / src.name

    print(f"📄 Source: {src}")
    print(f"📁 Destination: {dest}")

    if dest.exists():
        print("⚠️ Config already exists!")
        response = input("Overwrite? (y/N): ").strip().lower()
        if response != "y":
            print("❌ Aborted")
            sys.exit(0)

    shutil.copy2(src, dest)
    print("✅ Config copied successfully")


if __name__ == "__main__":
    main()
