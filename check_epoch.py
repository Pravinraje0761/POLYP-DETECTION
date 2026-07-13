
import torch
import os

checkpoint_path = "checkpoints/best_adenseunet.pth"
if os.path.exists(checkpoint_path):
    print(f"Loading checkpoint from {checkpoint_path}...")
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    print("\nCheckpoint contents:")
    for key in checkpoint.keys():
        print(f"  {key}: {checkpoint[key]}")
else:
    print(f"Checkpoint not found at {checkpoint_path}")

# Check for other checkpoint files
checkpoints_dir = "checkpoints"
if os.path.exists(checkpoints_dir):
    print(f"\nFiles in {checkpoints_dir}:")
    for file in os.listdir(checkpoints_dir):
        print(f"  {file}")
