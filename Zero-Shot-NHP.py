#!/usr/bin/env python3
"""
F5-TTS Zero-Shot Voice Cloning Script
Uses the official f5-tts_infer-cli command
"""

import subprocess
from pathlib import Path
import sys
import torch


class F5TTSVoiceCloner:
    def __init__(
        self,
        model_path: str,
        device: str = "cuda"
    ):
        """
        Initialize F5-TTS Voice Cloner
        
        Args:
            model_path: Path to F5-TTS model directory (standard or fine-tuned)
            device: Device to run on (cuda/cpu)
        """
        self.device = device
        self.model_path = Path(model_path)
        self.model_name = self.model_path.name
        
        if not self.model_path.exists():
            raise FileNotFoundError(f"Model path not found: {model_path}")
        
        print(f"\nModel: {self.model_name}")
        
        # Verify model files exist
        model_files = (
            list(self.model_path.glob("*.safetensors")) +
            list(self.model_path.glob("*.pt"))
        )
        
        if not model_files:
            raise FileNotFoundError(
                f"No model weights found in {self.model_path}"
            )
        
        print(f"Model initialized (using {self.model_path.name})")
    
    def synthesize(
        self,
        ref_audio_path: str,
        ref_text: str,
        gen_text: str,
        output_filename: str = "output.wav",
        speed: float = 1.0,
        temperature: float = 0.3
    ) -> str:
        """
        Generate speech with voice cloning
        
        Args:
            ref_audio_path: Path to reference audio (3-10 seconds recommended)
            ref_text: Transcription of reference audio (or path to .txt file)
            gen_text: Text to synthesize
            output_filename: Output filename (always saves to Outputs folder)
            speed: Speech rate multiplier
            
        Returns:
            Path to output audio file
        """
        ref_audio_path = Path(ref_audio_path)
        if not ref_audio_path.exists():
            raise FileNotFoundError(f"Reference audio not found: {ref_audio_path}")
        
        # If ref_text is a file path, read it
        if ref_text.endswith('.txt') or ref_text.endswith('.json'):
            text_file = Path(ref_text)
            if text_file.exists():
                with open(text_file, 'r') as f:
                    ref_text = f.read().strip()
        
        # Always save to Outputs folder
        output_dir = Path("Outputs")
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / output_filename
        
        print(f"\nVoice Cloning Synthesis")
        print(f"  Model: {self.model_name}")
        print(f"  Reference audio: {ref_audio_path.name}")
        print(f"  Reference text: '{ref_text}'")
        print(f"  Target text: '{gen_text}'")
        print(f"  Speed: {speed}x")
        print(f"  Temperature: {temperature}")
        print(f"  Device: {self.device}")
        
        print(f"\nGenerating audio (this may take 1-3 minutes)...")
        
        # Find the actual model checkpoint file
        model_file = None
        for ext in ['*.safetensors', '*.pt']:
            files = list(self.model_path.glob(ext))
            if files:
                model_file = files[0]
                break
        
        if not model_file:
            raise FileNotFoundError(f"No model checkpoint found in {self.model_path}")
        
        # Build command using official f5-tts_infer-cli command
        # Use -p/--ckpt_file for local fine-tuned models
        cmd = [
            "f5-tts_infer-cli",
            "-m", self.model_name,
            "-p", str(model_file),
            "-r", str(ref_audio_path.absolute()),
            "-s", ref_text,
            "-t", gen_text,
            "-o", str(output_dir),
            "-w", output_filename,
            "--speed", str(speed),
            "--device", self.device,
        ]
        
        print(f"Command: {' '.join(cmd)}\n")
        
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=604800)
            
            print(f"Return code: {result.returncode}")
            if result.stdout:
                print(f"Output:\n{result.stdout}")
            if result.stderr:
                print(f"Stderr:\n{result.stderr}")
            
            # Check if output file was created
            if output_path.exists():
                try:
                    # Try to get duration
                    wav, sr = torchaudio.load(str(output_path))
                    duration = wav.shape[-1] / sr
                    print(f"\n✓ Generated: {output_path} ({duration:.2f}s)")
                except:
                    print(f"\n✓ Generated: {output_path}")
                return str(output_path)
            else:
                raise FileNotFoundError(f"Output file not created: {output_path}")
        
        except subprocess.TimeoutExpired:
            print(f"Synthesis timed out after 10 minutes")
            raise
        except Exception as e:
            print(f"Error: {e}")
            raise


def list_available_models(base_dir: str = "Models/F5-TTS") -> list:
    """List available F5-TTS models in the base directory"""
    base_path = Path(base_dir)
    if not base_path.exists():
        return []
    
    models = []
    for item in base_path.iterdir():
        if item.is_dir():
            if any(item.glob("*.safetensors")) or any(item.glob("*.pt")):
                models.append(item)
    
    return sorted(models)


def interactive_mode():
    """Interactive command-line interface"""
    print("\n" + "="*60)
    print("F5-TTS Zero-Shot Voice Cloner")
    print("="*60)
    
    # Check CUDA
    print(f"\nDevice Status:")
    print(f"  CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"  GPU: {torch.cuda.get_device_name(0)}")
    
    # Model selection
    print("\nAvailable Models:")
    models = list_available_models()
    
    if not models:
        print("No models found in Models/F5-TTS")
        sys.exit(1)
    
    for i, model in enumerate(models, 1):
        print(f"  {i}. {model.name}")
    
    while True:
        try:
            choice = int(input(f"\nSelect model (1-{len(models)}): "))
            if 1 <= choice <= len(models):
                selected_model = models[choice - 1]
                break
        except ValueError:
            pass
        print(f"Please enter a number between 1 and {len(models)}")
    
    print(f"\n Using: {selected_model.name}")
    
    # Get reference audio path
    while True:
        ref_audio = input("\nReference audio file path: ").strip()
        if Path(ref_audio).exists():
            break
        print(f"File not found: {ref_audio}")
    
    # Get reference transcript
    ref_text = input("Reference audio transcript (or path to .txt file): ").strip()
    if not ref_text:
        print("Transcript cannot be empty")
        sys.exit(1)
    
    # Get text to generate
    print("\nText to generate:")
    print("  Option 1: Type text directly")
    print("  Option 2: Load from .txt file")
    text_choice = input("Choose (1 or 2): ").strip()
    
    if text_choice == "2":
        while True:
            gen_text_path = input("Path to .txt file: ").strip()
            if Path(gen_text_path).exists():
                with open(gen_text_path, 'r') as f:
                    gen_text = f.read().strip()
                break
            print(f"File not found: {gen_text_path}")
    else:
        gen_text = input("Type your text: ").strip()
    
    if not gen_text:
        print("Text cannot be empty")
        sys.exit(1)
    
    # Output filename (always saves to Outputs folder)
    output_filename = input("Output filename (default: output.wav): ").strip()
    if not output_filename:
        output_filename = "output.wav"
    
    # Speed
    speed_str = input("Speed (default 1.0, range 0.5-2.0): ").strip()
    speed = float(speed_str) if speed_str else 1.0
    
    # Temperature (controls hallucinations)
    temp_str = input("Temperature (default 0.3, range 0.1-1.0, lower = less hallucination): ").strip()
    temperature = float(temp_str) if temp_str else 0.3
    
    # Device selection
    has_cuda = torch.cuda.is_available()
    if has_cuda:
        device_str = input("Device (default cuda, options: cuda/cpu): ").strip().lower()
        device = device_str if device_str in ['cuda', 'cpu'] else 'cuda'
    else:
        print("CUDA not available, using CPU")
        device = 'cpu'
    
    # Initialize and synthesize
    try:
        cloner = F5TTSVoiceCloner(str(selected_model), device=device)
        cloner.synthesize(
            ref_audio_path=ref_audio,
            ref_text=ref_text,
            gen_text=gen_text,
            output_filename=output_filename,
            speed=speed,
            temperature=temperature
        )
        print("\nDone!")
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


def main():
    """Command-line interface"""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="F5-TTS Zero-Shot Voice Cloning",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Interactive mode (recommended)
  python voice_cloner.py
  
  # Direct mode with all arguments
  python voice_cloner.py \\
    --model "F5TTS_Fine_Tuned" \\
    --ref-audio "reference.wav" \\
    --ref-text "Hello, this is the reference" \\
    --gen-text "Generate this text" \\
    --output "my_output.wav"
        """
    )
    
    parser.add_argument("--model", help="Model directory name (e.g., F5TTS_Fine_Tuned)")
    parser.add_argument("--ref-audio", help="Path to reference audio file")
    parser.add_argument("--ref-text", help="Transcription of reference audio (or path to .txt)")
    parser.add_argument("--gen-text", help="Text to generate")
    parser.add_argument("--output", default="output.wav", help="Output filename (saves to Outputs folder)")
    parser.add_argument("--speed", type=float, default=1.0, help="Speech speed (0.5-2.0)")
    parser.add_argument("--temperature", type=float, default=0.3, help="Temperature for sampling (0.1-1.0, lower = less hallucination)")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu", help="Device (cuda/cpu)")
    
    args = parser.parse_args()
    
    required_args = [args.model, args.ref_audio, args.ref_text, args.gen_text]
    
    if all(required_args):
        # Direct mode
        try:
            model_dir = Path("Models/F5-TTS") / args.model
            cloner = F5TTSVoiceCloner(str(model_dir), device=args.device)
            cloner.synthesize(
                ref_audio_path=args.ref_audio,
                ref_text=args.ref_text,
                gen_text=args.gen_text,
                output_filename=args.output,
                speed=args.speed,
                temperature=args.temperature
            )
            print("\nDone!")
        except Exception as e:
            print(f"\nError: {e}")
            import traceback
            traceback.print_exc()
            sys.exit(1)
    else:
        # Interactive mode
        interactive_mode()


if __name__ == "__main__":
    main()