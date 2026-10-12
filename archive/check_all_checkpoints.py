
import os
import torch
import json

checkpoint_dir = "checkpoints"
print("Checking checkpoints in", checkpoint_dir)
for f in sorted(os.listdir(checkpoint_dir)):
    if not f.endswith(".pth"):
        continue
    print("\n", "="*60)
    print("File:", f)
    path = os.path.join(checkpoint_dir, f)
    try:
        ckpt = torch.load(path, map_location="cpu", weights_only=False)
        print("Keys in checkpoint:", list(ckpt.keys()))
        if "best_dice" in ckpt:
            print("best_dice:", ckpt["best_dice"])
        if "best_val_dice" in ckpt:
            print("best_val_dice:", ckpt["best_val_dice"])
        if "epoch" in ckpt:
            print("epoch:", ckpt["epoch"])
    except Exception as e:
        print("Error loading:", e)

split_path = os.path.join(checkpoint_dir, "ultimate_split_indices.json")
if os.path.exists(split_path):
    print("\n", "="*60)
    print("Split file:", split_path)
    with open(split_path, "r") as f:
        split = json.load(f)
    print("Train/Val/Test counts:", len(split["train"]), len(split["val"]), len(split["test"]))

