import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
import torch

ckpt_path = "outputs/tuned_pipeline/sseg_nase_tuned_pipeline/checkpoints/final_model.pt"
ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)

arch = ckpt.get("architecture_summary", None)
if arch is None:
    print("architecture_summary not found in checkpoint.")
else:
    print("architecture_summary:")
    for k, v in arch.items():
        print(f"  {k}: {v}")

print("\nKeys in checkpoint:", list(ckpt.keys()))
