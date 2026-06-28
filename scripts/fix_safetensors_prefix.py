from pathlib import Path

from safetensors.torch import load_file, save_file

SRC = Path("artifacts/hf_release/model.safetensors")
DST = SRC  # overwrite in place

print(f"Loading {SRC}...")
sd = load_file(SRC)
print(f"Loaded {len(sd)} keys")

# Add 'model.' prefix to all keys
new_sd = {}
for k, v in sd.items():
    new_k = f"model.{k}"
    new_sd[new_k] = v.contiguous()

print(f"First new key: {list(new_sd.keys())[0]}")
print(f"Saving back to {DST}...")
save_file(new_sd, DST)
print("Done.")
