
import os
import torch
import json

CHECKPOINT_DIR = "checkpoints"

print("="*100)
print("Checking all checkpoints:")
print("="*100)

all_checkpoints = [f for f in os.listdir(CHECKPOINT_DIR) if f.endswith(".pth")]

for filename in all_checkpoints:
    try:
        path = os.path.join(CHECKPOINT_DIR, filename)
        ckpt = torch.load(path, map_location="cpu", weights_only=False)
        epoch = ckpt.get("epoch", "N/A")
        best_dice = ckpt.get("best_dice", "N/A")
        print(f"{filename:50s} | Epoch: {epoch:5} | Best Dice: {best_dice:.6f}" if isinstance(best_dice, float) else f"{filename:50s} | Epoch: {epoch:5} | Best Dice: {best_dice}")
    except Exception as e:
        print(f"{filename:50s} | Error loading: {str(e)}")
