import torch
from pathlib import Path
from safetensors.torch import save_file

def convert_pt_to_safetensors(pt_file):
    """Convert PyTorch .pt file to safetensors format, handling nested dicts."""
    pt_path = Path(pt_file)
    output_path = pt_path.with_suffix('.safetensors')
    
    print(f"Loading {pt_path.name}...")
    checkpoint = torch.load(pt_path, map_location='cpu')
    
    # Handle nested structure - extract model_state_dict if it exists
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        model = checkpoint['model_state_dict']
        print(f"Extracted 'model_state_dict' from checkpoint")
    elif isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
        model = checkpoint['state_dict']
        print(f"Extracted 'state_dict' from checkpoint")
    else:
        model = checkpoint
    
    # Ensure all values are tensors
    if isinstance(model, dict):
        model = {k: v for k, v in model.items() if isinstance(v, torch.Tensor)}
    
    print(f"Saving to {output_path.name}...")
    save_file(model, str(output_path))
    print(f"✓ Conversion complete: {output_path}")
    return output_path

# Usage
if __name__ == "__main__":
    pt_file = "/home/jacob/Documents/GitHub/ML483-ViVoice/ZeroshotVoiceCloner_Tool/Models/F5-TTS/F5TTS_Fine_Tuned/model_1590.pt"
    convert_pt_to_safetensors(pt_file)