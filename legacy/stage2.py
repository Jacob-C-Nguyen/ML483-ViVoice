#!/usr/bin/env python3
"""
YourTTS Fine-tuning Script (Stage 2)
Fine-tune the base model on YOUR specific voice for best quality cloning
Run this AFTER Stage 1 completes
"""

import os
import torch
import soundfile as sf
import numpy as np
from pathlib import Path
import json
from dataclasses import dataclass
from typing import List, Dict
from tqdm import tqdm

# Install: pip install TTS torch torchaudio soundfile

from TTS.tts.configs.vits_config import VitsConfig
from TTS.tts.models.vits import Vits
from trainer import Trainer, TrainerArgs


@dataclass
class FinetuneConfig:
    """Configuration for fine-tuning on your voice"""
    # Paths
    base_model_path: str = "./yourtts_base_model"  # From Stage 1
    your_voice_data: str = "./my_voice_data"
    output_path: str = "./yourtts_my_voice"
    
    # Training (optimized for quick fine-tuning)
    batch_size: int = 4
    num_epochs: int = 50  # Much fewer epochs than base training
    learning_rate: float = 1e-5  # Lower LR for fine-tuning
    
    # Audio
    sample_rate: int = 22050
    min_audio_length: float = 2.0
    max_audio_length: float = 10.0


class VoiceDataProcessor:
    """Process your voice recordings for fine-tuning"""
    
    def __init__(self, config: FinetuneConfig):
        self.config = config
        self.data_path = Path(config.your_voice_data)
        self.data_path.mkdir(parents=True, exist_ok=True)
        (self.data_path / "wavs").mkdir(exist_ok=True)
        
    def process_audio_file(self, audio_path: Path, output_name: str) -> bool:
        """Process and validate audio file"""
        try:
            waveform, sr = sf.read(str(audio_path))
            
            # Convert to mono
            if waveform.ndim == 2:
                waveform = np.mean(waveform, axis=1)
            
            # Resample if needed
            if sr != self.config.sample_rate:
                from scipy import signal
                num_samples = int(len(waveform) * self.config.sample_rate / sr)
                waveform = signal.resample(waveform, num_samples)
            
            # Check duration
            duration = len(waveform) / self.config.sample_rate
            if duration < self.config.min_audio_length or duration > self.config.max_audio_length:
                return False
            
            # Save processed audio
            output_path = self.data_path / "wavs" / f"{output_name}.wav"
            sf.write(str(output_path), waveform, self.config.sample_rate)
            
            return True
            
        except Exception as e:
            print(f"Error processing {audio_path}: {e}")
            return False
    
    def create_dataset_from_folder(self, audio_folder: str, transcript_file: str = None):
        """
        Create dataset from your voice recordings
        
        Args:
            audio_folder: Folder with your voice .wav files
            transcript_file: Text file with transcripts (one per line)
        """
        audio_folder = Path(audio_folder)
        audio_files = sorted(list(audio_folder.glob("*.wav")) + list(audio_folder.glob("*.mp3")))
        
        if not audio_files:
            raise ValueError(f"No audio files found in {audio_folder}")
        
        print(f"Found {len(audio_files)} audio files")
        
        # Load transcripts
        transcripts = []
        if transcript_file and Path(transcript_file).exists():
            with open(transcript_file, 'r', encoding='utf-8') as f:
                transcripts = [line.strip() for line in f if line.strip()]
        
        # Process audio files
        metadata = []
        for idx, audio_path in enumerate(tqdm(audio_files, desc="Processing")):
            output_name = f"my_voice_{idx:04d}"
            
            if self.process_audio_file(audio_path, output_name):
                if idx < len(transcripts):
                    text = transcripts[idx]
                else:
                    text = f"Recording {idx}"
                
                metadata.append({
                    "audio_file": f"wavs/{output_name}.wav",
                    "text": text,
                    "speaker_name": "my_voice"
                })
        
        if not metadata:
            raise ValueError("No valid audio files were processed")
        
        # Save metadata
        metadata_path = self.data_path / "metadata.csv"
        with open(metadata_path, 'w', encoding='utf-8') as f:
            for item in metadata:
                f.write(f"{item['audio_file']}|{item['text']}|{item['speaker_name']}\n")
        
        print(f"✓ Dataset created: {len(metadata)} samples")
        print(f"✓ Metadata saved to: {metadata_path}")
        
        return len(metadata)
    
    def quick_guide(self):
        """Print guide for preparing your voice data"""
        print("\n" + "="*60)
        print("YOUR VOICE DATA PREPARATION GUIDE")
        print("="*60)
        print("\n1. Record 2-10 minutes of your voice:")
        print("   - Use good quality microphone")
        print("   - Quiet environment (no background noise)")
        print("   - Speak naturally and clearly")
        print("   - Vary your intonation and emotion")
        print("\n2. Split into clips of 3-10 seconds each")
        print("   - Aim for 20-100 clips")
        print("   - Each clip should be one complete sentence")
        print("\n3. Create a folder with your audio files:")
        print(f"   mkdir -p {self.data_path}/recordings")
        print("   # Place your .wav files there")
        print("\n4. Create transcripts (IMPORTANT for quality!):")
        print(f"   nano {self.data_path}/transcripts.txt")
        print("   # One line per audio file, in same order")
        print("\n5. Run this script and it will process everything!")
        print("\n" + "="*60 + "\n")


def finetune_formatter(root_path, meta_file, **kwargs):
    """Formatter for fine-tuning data"""
    items = []
    txt_file = os.path.join(root_path, meta_file)
    
    if not os.path.exists(txt_file):
        return items
    
    with open(txt_file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            
            parts = line.split("|")
            if len(parts) < 3:
                continue
            
            wav_file = os.path.join(root_path, parts[0])
            text = parts[1]
            speaker_name = parts[2]
            
            if not os.path.exists(wav_file):
                continue
            
            items.append({
                "text": text,
                "audio_file": wav_file,
                "speaker_name": speaker_name,
                "root_path": root_path,
            })
    
    return items


def main():
    """Main fine-tuning pipeline - Stage 2"""
    
    print("="*60)
    print("YourTTS FINE-TUNING (Stage 2)")
    print("Fine-tune base model on YOUR voice")
    print("="*60)
    
    # Configuration
    config = FinetuneConfig(
        base_model_path="./yourtts_base_model",
        your_voice_data="./my_voice_data",
        output_path="./yourtts_my_voice"
    )
    
    # Check base model exists
    base_checkpoint = Path(config.base_model_path) / "best_model.pth"
    if not base_checkpoint.exists():
        # Try latest checkpoint
        checkpoints = list(Path(config.base_model_path).glob("checkpoint_*.pth"))
        if checkpoints:
            base_checkpoint = sorted(checkpoints)[-1]
        else:
            print(f"\n❌ ERROR: No base model found at {config.base_model_path}")
            print("Please run Stage 1 first to train the base model!")
            return
    
    print(f"\n✓ Found base model: {base_checkpoint}")
    
    # Step 1: Prepare your voice data
    print("\n" + "="*60)
    print("Step 1: Preparing Your Voice Data")
    print("="*60)
    
    processor = VoiceDataProcessor(config)
    
    metadata_path = processor.data_path / "metadata.csv"
    if not metadata_path.exists():
        processor.quick_guide()
        
        # Check if user has audio folder ready
        recordings_folder = processor.data_path / "recordings"
        transcripts_file = processor.data_path / "transcripts.txt"
        
        if recordings_folder.exists() and any(recordings_folder.glob("*.wav")):
            print("Found recordings folder! Processing...")
            num_samples = processor.create_dataset_from_folder(
                str(recordings_folder),
                str(transcripts_file) if transcripts_file.exists() else None
            )
        else:
            print("\n⚠️  No recordings found yet!")
            print("Follow the guide above to prepare your voice data, then run again.")
            return
    else:
        with open(metadata_path, 'r') as f:
            num_samples = sum(1 for line in f)
        print(f"✓ Found existing dataset with {num_samples} samples")
    
    if num_samples < 10:
        print(f"\n⚠️  WARNING: Only {num_samples} samples!")
        print("Recommend 20-100 samples for best quality.")
        response = input("Continue anyway? (y/n): ")
        if response.lower() != 'y':
            return
    
    # Register formatter
    import TTS.tts.datasets as datasets_module
    import TTS.tts.datasets.formatters as formatters
    datasets_module.finetune_voice = finetune_formatter
    formatters.finetune_voice = finetune_formatter
    
    # Step 2: Load base model config
    print("\n" + "="*60)
    print("Step 2: Loading Base Model Configuration")
    print("="*60)
    
    base_config_path = Path(config.base_model_path) / "config.json"
    if not base_config_path.exists():
        print(f"❌ ERROR: Config not found at {base_config_path}")
        return
    
    # Load and modify config for fine-tuning
    finetune_config = VitsConfig()
    finetune_config.load_json(str(base_config_path))
    
    # Override for fine-tuning
    finetune_config.output_path = config.output_path
    finetune_config.batch_size = config.batch_size
    finetune_config.epochs = config.num_epochs
    finetune_config.lr = config.learning_rate
    finetune_config.save_step = 100  # Save more frequently
    finetune_config.print_step = 25
    
    # Update dataset to your voice
    finetune_config.datasets = [{
        "formatter": "finetune_voice",
        "dataset_name": "my_voice",
        "path": str(processor.data_path),
        "meta_file_train": "metadata.csv",
        "meta_file_val": "metadata.csv",
        "language": "en",
    }]
    
    os.makedirs(config.output_path, exist_ok=True)
    finetune_config.save_json(os.path.join(config.output_path, "config.json"))
    print("✓ Fine-tuning configuration ready")
    
    # Step 3: Load base model
    print("\n" + "="*60)
    print("Step 3: Loading Base Model")
    print("="*60)
    
    try:
        model = Vits.init_from_config(finetune_config)
        
        # Load base model weights
        print(f"Loading weights from: {base_checkpoint}")
        checkpoint = torch.load(base_checkpoint, map_location="cpu")
        model.load_state_dict(checkpoint["model"])
        
        model.cuda()
        print("✓ Base model loaded successfully")
        
    except Exception as e:
        print(f"❌ ERROR loading base model: {e}")
        return
    
    # Step 4: Setup trainer
    print("\n" + "="*60)
    print("Step 4: Setting up Fine-tuning Trainer")
    print("="*60)
    
    trainer_args = TrainerArgs()
    
    trainer = Trainer(
        trainer_args,
        finetune_config,
        config.output_path,
        model=model,
    )
    
    print("✓ Trainer initialized")
    
    # Step 5: Start fine-tuning
    print("\n" + "="*60)
    print("Step 5: Fine-tuning on Your Voice")
    print("="*60)
    print(f"\nFine-tuning with {num_samples} samples of YOUR voice")
    print(f"Checkpoints: {config.output_path}")
    print(f"Monitor: tensorboard --logdir {config.output_path}")
    print(f"\n⏱️  Estimated time: 1-2 hours for {config.num_epochs} epochs")
    print("🎯 Best quality usually appears after 20-30 epochs")
    print("\nPress Ctrl+C to stop early if quality is good\n")
    
    try:
        trainer.fit()
        
        print("\n" + "="*60)
        print("✓ STAGE 2 COMPLETE - Your Voice Model Ready!")
        print("="*60)
        print(f"\nPersonalized model saved in: {config.output_path}")
        print("\n🎉 You now have a model optimized for YOUR voice!")
        print("\nTo use it:")
        print("from TTS.api import TTS")
        print(f"tts = TTS(model_path='{config.output_path}/best_model.pth')")
        print("tts.tts_to_file(text='Hello!', file_path='output.wav')")
        
    except KeyboardInterrupt:
        print("\n\nFine-tuning interrupted.")
        print(f"Checkpoints saved in: {config.output_path}")
        
    except Exception as e:
        print(f"\n❌ Error: {e}")


if __name__ == "__main__":
    # Install dependencies
    try:
        import scipy
    except ImportError:
        os.system("pip install scipy")
    
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB\n")
    
    main()
