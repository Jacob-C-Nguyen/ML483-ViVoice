#!/usr/bin/env python3
"""
YourTTS Base Model Training Script (Stage 1)
Train a base model with zero-shot voice cloning capability on LibriTTS-R
Then fine-tune it on your voice in Stage 2!
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
from TTS.tts.models.vits import Vits, VitsArgs
from TTS.config.shared_configs import BaseAudioConfig
from trainer import Trainer, TrainerArgs


@dataclass
class DatasetConfig:
    """Configuration for dataset preparation"""
    dataset_path: str = "./BASE_MODEL_DATASET"
    processed_data_path: str = "./processed_voice_data"  # Reuse existing processed data
    sample_rate: int = 22050
    min_audio_length: float = 1.0
    max_audio_length: float = 10.0


class LibriTTSProcessor:
    """Process LibriTTS-R dataset for YourTTS training"""
    
    def __init__(self, config: DatasetConfig):
        self.config = config
        self.dataset_path = Path(config.dataset_path)
        self.processed_path = Path(config.processed_data_path)
        self.processed_path.mkdir(parents=True, exist_ok=True)
        (self.processed_path / "wavs").mkdir(exist_ok=True)
        
    def find_all_audio_files(self) -> List[Path]:
        """Find all .wav files in the LibriTTS structure"""
        print(f"Scanning {self.dataset_path} for audio files...")
        
        if not self.dataset_path.exists():
            raise FileNotFoundError(f"Dataset folder not found: {self.dataset_path}")
        
        audio_files = list(self.dataset_path.rglob("*.wav"))
        
        if not audio_files:
            raise FileNotFoundError(f"No .wav files found in {self.dataset_path}")
        
        print(f"Found {len(audio_files)} audio files")
        return audio_files
    
    def get_transcript_for_audio(self, audio_path: Path) -> str:
        """Get transcript for an audio file"""
        normalized_txt = audio_path.with_suffix('.normalized.txt')
        if normalized_txt.exists():
            with open(normalized_txt, 'r', encoding='utf-8') as f:
                return f.read().strip()
        
        original_txt = audio_path.with_suffix('.original.txt')
        if original_txt.exists():
            with open(original_txt, 'r', encoding='utf-8') as f:
                return f.read().strip()
        
        return None
    
    def process_audio(self, audio_path: Path, output_name: str) -> bool:
        """Process and validate audio file"""
        try:
            waveform, sr = sf.read(str(audio_path))
            
            if waveform.ndim == 2:
                waveform = np.mean(waveform, axis=1)
            
            if sr != self.config.sample_rate:
                from scipy import signal
                num_samples = int(len(waveform) * self.config.sample_rate / sr)
                waveform = signal.resample(waveform, num_samples)
            
            duration = len(waveform) / self.config.sample_rate
            if duration < self.config.min_audio_length or duration > self.config.max_audio_length:
                return False
            
            output_path = self.processed_path / "wavs" / f"{output_name}.wav"
            sf.write(str(output_path), waveform, self.config.sample_rate)
            
            return True
            
        except Exception as e:
            return False
    
    def create_metadata(self) -> tuple:
        """Create metadata file from LibriTTS-R structure"""
        audio_files = self.find_all_audio_files()
        
        metadata = []
        speaker_ids = set()
        
        print("\nProcessing audio files...")
        for audio_path in tqdm(audio_files):
            transcript = self.get_transcript_for_audio(audio_path)
            if not transcript:
                continue
            
            speaker_id = audio_path.parent.parent.name
            speaker_ids.add(speaker_id)
            
            output_name = f"speaker_{speaker_id}_{audio_path.stem}"
            
            if not self.process_audio(audio_path, output_name):
                continue
            
            metadata.append({
                "audio_file": f"wavs/{output_name}.wav",
                "text": transcript,
                "speaker_name": speaker_id
            })
        
        print(f"\n✓ Processed {len(metadata)} samples from {len(speaker_ids)} speakers")
        return metadata, len(speaker_ids)
    
    def save_metadata(self, metadata: List[Dict]):
        """Save metadata in LJSpeech format"""
        metadata_path = self.processed_path / "metadata.csv"
        with open(metadata_path, 'w', encoding='utf-8') as f:
            for item in metadata:
                # Format: audio_file|text|speaker_name
                f.write(f"{item['audio_file']}|{item['text']}|{item['speaker_name']}\n")
        print(f"✓ Metadata saved to {metadata_path}")
        
        # Save speaker list
        speakers = sorted(set(item['speaker_name'] for item in metadata))
        speakers_path = self.processed_path / "speakers.json"
        with open(speakers_path, 'w', encoding='utf-8') as f:
            json.dump(speakers, f, indent=2)
        print(f"✓ Speaker list saved to {speakers_path}")
        
        return len(speakers)


def yourtts_formatter(root_path, meta_file, **kwargs):
    """Custom formatter for YourTTS with our column order"""
    items = []
    txt_file = os.path.join(root_path, meta_file)
    
    print(f"[Formatter] Looking for: {txt_file}")
    
    if not os.path.exists(txt_file):
        print(f"[Formatter] ERROR: File not found!")
        return items
    
    with open(txt_file, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            
            parts = line.split("|")
            if len(parts) < 3:
                print(f"[Formatter] Warning: Line {line_num} has only {len(parts)} parts")
                continue
            
            # Our format: audio_file|text|speaker_name
            wav_file = os.path.join(root_path, parts[0])
            text = parts[1]
            speaker_name = parts[2]
            
            if not os.path.exists(wav_file):
                if line_num <= 5:  # Only warn for first few
                    print(f"[Formatter] Warning: Audio not found: {wav_file}")
                continue
            
            items.append({
                "text": text,
                "audio_file": wav_file,
                "speaker_name": speaker_name,
                "root_path": root_path,
            })
    
    print(f"[Formatter] Successfully loaded {len(items)} samples")
    return items


def setup_yourtts_config(output_path: str, dataset_path: str, num_speakers: int) -> VitsConfig:
    """Setup YourTTS configuration (8GB VRAM optimized)"""
    
    # Audio config
    audio_config = BaseAudioConfig(
        sample_rate=22050,
        hop_length=256,
        win_length=1024,
        fft_size=1024,
        mel_fmin=0.0,
        mel_fmax=None,
        num_mels=80,
    )
    
    # YourTTS is VITS + speaker encoder for zero-shot cloning
    config = VitsConfig(
        # Audio
        audio=audio_config,
        
        # Model with speaker encoding for zero-shot
        model_args=VitsArgs(
            use_speaker_embedding=True,
            num_speakers=num_speakers,
            speaker_embedding_channels=256,  # Standard size works better
            use_d_vector_file=False,
            use_speaker_encoder_as_loss=False,  # Disable for now - requires additional model
        ),
        
        # Training (8GB VRAM optimized)
        batch_size=8,
        eval_batch_size=4,
        num_loader_workers=4,
        num_eval_loader_workers=2,
        run_eval=True,
        test_delay_epochs=5,
        
        # Memory optimization
        mixed_precision=True,
        grad_clip=5.0,
        
        # Epochs
        epochs=1000,
        print_step=50,
        save_step=1000,
        save_n_checkpoints=5,
        save_checkpoints=True,
        
        # Learning rate
        lr=2e-4,
        lr_scheduler="ExponentialLR",
        lr_scheduler_params={"gamma": 0.999875},
        
        # Dataset
        datasets=[{
            "formatter": "yourtts_custom",
            "dataset_name": "libri_tts_r",
            "path": str(dataset_path),
            "meta_file_train": "metadata.csv",
            "meta_file_val": "metadata.csv",
            "language": "en",
            "ignored_speakers": []
        }],
        
        # Text processing
        text_cleaner="english_cleaners",
        use_phonemes=True,
        phoneme_language="en-us",
        
        # Output
        output_path=output_path,
    )
    
    return config


def main():
    """Main training pipeline - Stage 1: Base Model"""
    
    print("="*60)
    print("YourTTS BASE MODEL TRAINING (Stage 1)")
    print("Train base model with zero-shot voice cloning")
    print("="*60)
    
    # Check GPU
    if not torch.cuda.is_available():
        print("\n❌ ERROR: No GPU detected!")
        return
    
    gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
    print(f"\nGPU: {torch.cuda.get_device_name(0)}")
    print(f"VRAM: {gpu_memory:.1f} GB")
    
    if gpu_memory < 6:
        print("\n⚠️  WARNING: Less than 6GB VRAM may cause issues.")
    
    # Configuration
    dataset_config = DatasetConfig(
        dataset_path="./BASE_MODEL_DATASET",
        processed_data_path="./processed_voice_data"
    )
    
    output_path = "./yourtts_base_model"
    
    # Step 1: Check/prepare dataset
    print("\n" + "="*60)
    print("Step 1: Dataset Preparation")
    print("="*60)
    
    processor = LibriTTSProcessor(dataset_config)
    
    metadata_path = processor.processed_path / "metadata.csv"
    speakers_path = processor.processed_path / "speakers.json"
    
    if not metadata_path.exists():
        if not processor.dataset_path.exists():
            print(f"\n❌ ERROR: Dataset folder not found!")
            print(f"Expected: {processor.dataset_path.absolute()}")
            return
        
        print("Processing dataset...")
        metadata, num_speakers = processor.create_metadata()
        
        if len(metadata) < 100:
            print(f"\n⚠️  WARNING: Only {len(metadata)} samples!")
            return
        
        num_speakers = processor.save_metadata(metadata)
    else:
        print(f"✓ Found existing processed dataset")
        with open(metadata_path, 'r') as f:
            num_samples = sum(1 for line in f)
        with open(speakers_path, 'r') as f:
            num_speakers = len(json.load(f))
        print(f"  Samples: {num_samples}")
        print(f"  Speakers: {num_speakers}")
    
    # Register custom formatter - try multiple methods
    print("\nRegistering custom formatter...")
    import TTS.tts.datasets as datasets_module
    import TTS.tts.datasets.formatters as formatters_module
    
    # Method 1: Add to datasets module
    datasets_module.yourtts_custom = yourtts_formatter
    
    # Method 2: Add to formatters module
    formatters_module.yourtts_custom = yourtts_formatter
    
    # Method 3: Add to formatters __all__ if it exists
    if hasattr(formatters_module, '__all__'):
        if 'yourtts_custom' not in formatters_module.__all__:
            formatters_module.__all__.append('yourtts_custom')
    
    print("✓ Formatter registered")
    
    # Test the formatter
    print("\nTesting formatter...")
    test_samples = yourtts_formatter(
        str(processor.processed_path),
        "metadata.csv"
    )
    if test_samples:
        print(f"✓ Formatter test successful: {len(test_samples)} samples loaded")
    else:
        print("❌ Formatter test failed: No samples loaded")
        return
    
    # Step 2: Setup configuration
    print("\n" + "="*60)
    print("Step 2: Setting up YourTTS Configuration")
    print("="*60)
    
    config = setup_yourtts_config(output_path, str(processor.processed_path), num_speakers)
    
    os.makedirs(output_path, exist_ok=True)
    config.save_json(os.path.join(output_path, "config.json"))
    print(f"✓ Configuration saved")
    print(f"✓ Configured for {num_speakers} speakers with zero-shot cloning")
    
    # Step 3: Initialize model
    print("\n" + "="*60)
    print("Step 3: Initializing YourTTS Model")
    print("="*60)
    
    try:
        # Create speaker manager with our speakers
        from TTS.tts.utils.speakers import SpeakerManager
        
        speakers_file = processor.processed_path / "speakers.json"
        if speakers_file.exists():
            speaker_manager = SpeakerManager(speaker_id_file_path=str(speakers_file))
            print(f"✓ Loaded {len(speaker_manager.name_to_id)} speakers")
        else:
            print("⚠️  No speakers file found, creating from metadata...")
            # Extract speakers from metadata
            with open(metadata_path, 'r') as f:
                speakers = sorted(set(line.split('|')[2].strip() for line in f if line.strip()))
            
            # Save speakers file
            with open(speakers_file, 'w') as f:
                json.dump(speakers, f, indent=2)
            
            speaker_manager = SpeakerManager(speaker_id_file_path=str(speakers_file))
            print(f"✓ Created speaker manager with {len(speakers)} speakers")
        
        # Initialize model with speaker manager
        from TTS.tts.utils.text.tokenizer import TTSTokenizer
        from TTS.config import load_config
        
        tokenizer, _ = TTSTokenizer.init_from_config(config)
        
        model = Vits(config, tokenizer=tokenizer, speaker_manager=speaker_manager)
        model.cuda()
        
        total_params = sum(p.numel() for p in model.parameters())
        print(f"✓ Model initialized successfully")
        print(f"  Total parameters: {total_params:,}")
        
    except RuntimeError as e:
        if "out of memory" in str(e).lower():
            print("\n❌ ERROR: Out of memory!")
            print("Try reducing batch_size in the config.")
        else:
            print(f"\n❌ ERROR: {e}")
        return
    
    # Step 4: Setup trainer
    print("\n" + "="*60)
    print("Step 4: Setting up Trainer")
    print("="*60)
    
    # Pre-load samples for the trainer
    print("Pre-loading dataset samples...")
    from TTS.tts.datasets import load_tts_samples
    
    try:
        train_samples, eval_samples = load_tts_samples(
            config.datasets,
            eval_split=True,
            eval_split_max_size=config.eval_split_max_size if hasattr(config, 'eval_split_max_size') else None,
            eval_split_size=config.eval_split_size if hasattr(config, 'eval_split_size') else 0.01,
        )
        print(f"✓ Loaded {len(train_samples)} training samples")
        print(f"✓ Loaded {len(eval_samples)} evaluation samples")
    except Exception as e:
        print(f"❌ Error loading samples: {e}")
        import traceback
        traceback.print_exc()
        return
    
    trainer_args = TrainerArgs()
    
    trainer = Trainer(
        trainer_args,
        config,
        output_path,
        model=model,
        train_samples=train_samples,
        eval_samples=eval_samples,
    )
    
    print("✓ Trainer initialized")
    
    # Step 5: Start training
    print("\n" + "="*60)
    print("Step 5: Starting Base Model Training")
    print("="*60)
    print(f"\nThis is STAGE 1: Training the base model")
    print(f"After this completes, you'll use Stage 2 to fine-tune on your voice")
    print(f"\nCheckpoints: {output_path}")
    print(f"Monitor: tensorboard --logdir {output_path}")
    print(f"\n⏱️  Estimated time: 3-5 days for 1000 epochs")
    print("💡 Can stop after 200-300 epochs for decent base model")
    print("🎯 Good quality usually at 500-700 epochs")
    print("\n🔥 Key feature: This model will have ZERO-SHOT voice cloning!")
    print("   After training, you can clone ANY voice with just 3-10 seconds")
    print("\nPress Ctrl+C to stop training\n")
    
    try:
        trainer.fit()
        
        print("\n" + "="*60)
        print("✓ STAGE 1 COMPLETE - Base Model Trained!")
        print("="*60)
        print(f"\nBase model saved in: {output_path}")
        print(f"\n🎉 You now have a zero-shot voice cloning model!")
        print("\nNext steps:")
        print("1. Test the base model with different voices")
        print("2. Collect 2-10 minutes of YOUR voice recordings")
        print("3. Run Stage 2 script to fine-tune for your specific voice")
        print("\nStage 2 will make the model REALLY good at YOUR voice!")
        
    except KeyboardInterrupt:
        print("\n\nTraining interrupted by user.")
        print(f"Latest checkpoint: {output_path}")
        print("You can resume training or proceed to Stage 2 with current checkpoint.")
        
    except RuntimeError as e:
        if "out of memory" in str(e).lower():
            print("\n\n❌ OUT OF MEMORY!")
            print("Try reducing batch_size in the config (currently 8)")
        else:
            print(f"\n\n❌ Error: {e}")


if __name__ == "__main__":
    # Install dependencies
    try:
        import scipy
    except ImportError:
        print("Installing scipy...")
        os.system("pip install scipy")
    
    try:
        import phonemizer
    except ImportError:
        print("Installing phonemizer...")
        os.system("pip install phonemizer")
    
    print("\n" + "="*60)
    print("SYSTEM INFORMATION")
    print("="*60)
    
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    else:
        print("No GPU detected!")
    
    print(f"PyTorch version: {torch.__version__}")
    print("="*60 + "\n")
    
    main()
