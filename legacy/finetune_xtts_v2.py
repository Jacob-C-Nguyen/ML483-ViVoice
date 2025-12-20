#!/usr/bin/env python3
"""
finetune_xtts_v2.py

Fine-tune XTTS_v2 on a prepared dataset folder.

Expect dataset layout:
  dataset/
    metadata.csv      (pipe '|' or comma separated: filename|text|lang|speaker or filename|text)
    clips/
      *.wav

Usage examples:

# Auto-find xtts_v2 in local cache and fine-tune on GPU 0
python finetune_xtts_v2.py --dataset_dir ./dataset --output_dir ./xtts_finetune_out --epochs 20 --batch_size 16 --cuda_devices "0"

# Use explicit restore checkpoint (if you downloaded a base model manually)
python finetune_xtts_v2.py --dataset_dir ./dataset --output_dir ./xtts_finetune_out --restore_path /path/to/model.pth

#ON MY STUFF: 
#   MAKE SURE YOU ARE IN THE RIGHT ENVIRONMENT: 
#       
#   MAKE SURE YOU ARE IN "/home/jacob/Documents/GitHub/ML483-ViVoice/baseModelMaker"
#   python3 finetune_xtts_v2.py --dataset_dir dataset --output_dir OUTPUT_DIR_XTTS_V2 --epochs 20 --batch_size 22 --cuda_devices "0"

# Force CPU-only
python finetune_xtts_v2.py --dataset_dir ./dataset --output_dir ./xtts_finetune_out --cuda_devices ""
"""
import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
import soundfile as sf

# Typical TTS cache dirs to attempt to find xtts_v2
TTS_CACHE_DIRS = [
    Path.home() / ".local" / "share" / "tts",
    Path.home() / ".cache" / "tts",
    Path.home() / ".local" / "share" / "TTS",
    Path.home() / ".cache" / "TTS",
]

DEFAULT_SAMPLE_RATE = 24000  # xtts_v2 uses 24k


def pick_checkpoint_from_modeldir(model_dir: Path):
    """Return path to best candidate checkpoint or None."""
    candidates = sorted(model_dir.glob("checkpoint_*.pth"), key=lambda x: x.stat().st_mtime, reverse=True)
    if candidates:
        return str(candidates[0])
    for name in ("best_model.pth", "model.pth"):
        p = model_dir / name
        if p.exists():
            return str(p)
    return None

def validate_dataset(dataset_dir: Path):
    """Check basic dataset layout and sample rate consistency (warn only)."""
    if not dataset_dir.exists():
        raise FileNotFoundError(f"Dataset dir not found: {dataset_dir}")
    meta = dataset_dir / "metadata.csv"
    clips = dataset_dir / "clips"
    if not meta.exists():
        raise FileNotFoundError(f"metadata.csv not found in dataset dir: {meta}")
    if not clips.exists() or not any(clips.glob("*.wav")):
        raise FileNotFoundError(f"No .wav files found under: {clips}")
    # sample rate check (inspect first wav)
    first_wav = next(clips.glob("*.wav"))
    info = sf.info(str(first_wav))
    sr = info.samplerate
    if sr != DEFAULT_SAMPLE_RATE:
        print(f"WARNING: first clip sample rate is {sr} Hz but XTTS_v2 expects {DEFAULT_SAMPLE_RATE} Hz.")
        print("You should resample to 24000 Hz for best results. Continuing anyway...")

def write_config(cfg: dict, out_path: Path):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(cfg, fh, indent=2, ensure_ascii=False)
    print("Wrote config:", out_path)

def build_config(args, base_cfg: dict | None):
    # Start from base config if available, else create minimal
    cfg = dict(base_cfg) if base_cfg else {}
    # Safe top-level overrides:
    cfg["model"] = "xtts"
    cfg["run_name"] = args.run_name
    cfg["run_description"] = f"Fine-tune xtts_v2 on {args.dataset_dir}"
    cfg["output_path"] = args.output_dir
    cfg["epochs"] = args.epochs
    cfg["batch_size"] = args.batch_size
    cfg["eval_batch_size"] = max(1, args.batch_size // 4)
    cfg["num_loader_workers"] = args.num_workers
    cfg["num_eval_loader_workers"] = max(1, args.num_workers // 2)
    cfg["save_step"] = args.save_step
    cfg["log_model_step"] = args.log_model_step
    cfg["save_n_checkpoints"] = args.save_n_checkpoints
    cfg["mixed_precision"] = args.mixed_precision
    cfg["precision"] = "fp16" if args.mixed_precision else "fp32"
    cfg["run_eval"] = args.run_eval

    # audio
    if "audio" not in cfg:
        cfg["audio"] = {}
    cfg["audio"]["sample_rate"] = args.sample_rate
    cfg["audio"]["num_mels"] = cfg["audio"].get("num_mels", 80)
    cfg["audio"]["fft_size"] = cfg["audio"].get("fft_size", 1024)
    cfg["audio"]["hop_length"] = cfg["audio"].get("hop_length", 256)
    cfg["audio"]["win_length"] = cfg["audio"].get("win_length", cfg["audio"]["fft_size"])

    # dataset
    cfg["datasets"] = [{
        "formatter": args.formatter,
        "path": str(Path(args.dataset_dir).resolve()),
        "meta_file_train": args.meta_file,
        "meta_file_val": args.meta_file,   # using same file; trainer will split if needed
        "language": args.language
    }]

    # other sensible defaults
    cfg["text_cleaner"] = cfg.get("text_cleaner", "multilingual_cleaners")
    cfg["use_phonemes"] = args.use_phonemes
    if args.phoneme_language:
        cfg["phoneme_language"] = args.phoneme_language

    return cfg

def call_training(config_path: str, restore_path: str | None, cuda_devices: str):
    # Explicit path to your local train_tts.py
    local_train = Path("/home/jacob/Documents/GitHub/ML483-ViVoice/baseModelMaker/train_gpt_xtts.py")

    if local_train.exists():
        cmd = [sys.executable, str(local_train), "--config_path", str(config_path)]
        if restore_path:
            cmd += ["--restore_path", str(restore_path)]
        print("Using LOCAL train_tts.py:", local_train)
    else:
        # fallback to the installed CLI
        cmd = [
            "tts",
            "--config_path", str(config_path)
        ]
        if restore_path:
            cmd += ["--model_path", str(restore_path)]
        print("WARNING: local train_tts.py not found, using 'tts' CLI.")

    # Set compute device
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = cuda_devices

    print("Launching training:", " ".join(cmd))
    proc = subprocess.Popen(cmd, env=env)
    proc.wait()
    return proc.returncode


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset_dir", required=True, help="Path to dataset folder (contains metadata.csv and clips/)")
    p.add_argument("--meta_file", default="metadata.csv", help="metadata filename inside dataset_dir")
    p.add_argument("--output_dir", default="./xtts_finetune_out", help="where to save outputs")
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--restore_path", default=None, help="optional checkpoint .pth to restore from")
    p.add_argument("--run_name", default="xtts_v2_finetune")
    p.add_argument("--save_step", type=int, default=2000)
    p.add_argument("--log_model_step", type=int, default=1000)
    p.add_argument("--save_n_checkpoints", type=int, default=3)
    p.add_argument("--mixed_precision", action="store_true", help="use FP16 mixed precision")
    p.add_argument("--cuda_devices", default="0", help='CUDA_VISIBLE_DEVICES value (set "" to force CPU)')
    p.add_argument("--sample_rate", type=int, default=DEFAULT_SAMPLE_RATE, help="target sample rate (Hz) for model")
    p.add_argument("--formatter", default="ljspeech", help="dataset formatter (ljspeech common for metadata.csv)")
    p.add_argument("--language", default="en", help="language code")
    p.add_argument("--use_phonemes", action="store_true")
    p.add_argument("--phoneme_language", default="en-us")
    p.add_argument("--run_eval", action="store_true", help="enable evaluation during training (slower)")
    args = p.parse_args()

    dataset_dir = Path(args.dataset_dir)
    try:
        validate_dataset(dataset_dir)
    except Exception as e:
        print("Dataset validation error:", e)
        sys.exit(1)

    # ====== FORCE USE OF LOCAL XTTS MODEL ======
    # Your folder name is XTTS-v2, but model type is "xtts"
    model_dir = Path("pre-trained/XTTS-v2")

    if not model_dir.exists():
        print("ERROR: Local XTTS directory not found:", model_dir)
        sys.exit(1)

    cfg_file = model_dir / "config.json"
    restore_path = model_dir / "model.pth"

    if not cfg_file.exists():
        print("ERROR: Missing config.json in local XTTS folder.")
        sys.exit(1)

    if not restore_path.exists():
        print("ERROR: Missing model.pth in local XTTS folder.")
        sys.exit(1)

    print("Using local XTTS model at:", model_dir)
    print("Using restore checkpoint:", restore_path)

    with open(cfg_file, "r") as fh:
        base_cfg = json.load(fh)

    # Coqui TTS 0.22.0 only recognizes `xtts`
    # HuggingFace models use `xtts_v2` which breaks training
    base_cfg["model"] = "xtts"
    base_cfg.pop("model_type", None)


    finetune_cfg = build_config(args, base_cfg)

    # write config file
    config_path = Path.cwd() / "xtts_finetune_config.json"
    write_config(finetune_cfg, config_path)

    # confirm and run
    print("\n--- READY TO LAUNCH ---")
    print("dataset:", dataset_dir)
    print("restore_path:", restore_path)
    print("output_dir:", args.output_dir)
    print("epochs:", args.epochs, "batch_size:", args.batch_size)
    print("sample_rate:", args.sample_rate)
    print("-----------------------\n")

    ret = call_training(str(config_path), restore_path, args.cuda_devices)
    if ret == 0:
        print("Training finished successfully.")
    else:
        print("Training exited with code", ret)
        sys.exit(ret)

if __name__ == "__main__":
    main()
