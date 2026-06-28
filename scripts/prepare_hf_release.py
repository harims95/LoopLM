import json
import shutil
from pathlib import Path

import torch
from safetensors.torch import save_file

ART = Path("artifacts")
OUT = ART / "hf_release"
OUT.mkdir(exist_ok=True)

# Load checkpoint
print(f"Loading {ART / 'ckpt_17500.pt'}...")
ckpt = torch.load(ART / "ckpt_17500.pt", map_location="cpu", weights_only=False)
print(f"Checkpoint keys: {list(ckpt.keys())[:10]}")

# Extract model state dict
if "model" in ckpt:
    state_dict = ckpt["model"]
elif "state_dict" in ckpt:
    state_dict = ckpt["state_dict"]
else:
    # Assume the checkpoint IS the state dict
    state_dict = ckpt

# Strip "module." prefix if from DDP
clean_sd = {}
seen_ptrs = {}
for k, v in state_dict.items():
    new_k = k.replace("module.", "", 1) if k.startswith("module.") else k
    # Also strip _orig_mod. if from torch.compile
    new_k = new_k.replace("_orig_mod.", "", 1) if new_k.startswith("_orig_mod.") else new_k
    tensor = v.contiguous().cpu()
    ptr = tensor.untyped_storage().data_ptr()
    if ptr in seen_ptrs:
        # Break shared storage for tied weights so safetensors can serialize them.
        tensor = tensor.clone()
    else:
        seen_ptrs[ptr] = new_k
    clean_sd[new_k] = tensor

# Count params
total_params = sum(v.numel() for v in clean_sd.values())
print(f"Total params: {total_params:,} ({total_params/1e6:.1f}M)")

# Save as safetensors
save_file(clean_sd, OUT / "model.safetensors")
size_mb = (OUT / "model.safetensors").stat().st_size / (1024**2)
print(f"Saved model.safetensors: {size_mb:.1f} MB")

# Copy metadata
shutil.copy(ART / "config.json", OUT / "config.json")
shutil.copy(ART / "spec.json", OUT / "spec.json")

# Print first few keys for verification
print("\nFirst 10 weight names:")
for k in list(clean_sd.keys())[:10]:
    print(f"  {k}: {tuple(clean_sd[k].shape)}")

print(f"\nDone. Release files in {OUT}/")
