
"""
Script to check the latest model accuracies from all checkpoint files
"""
import os
import torch

CHECKPOINT_DIR = "checkpoints"

print("=" * 80)
print("LATEST MODEL ACCURACIES")
print("=" * 80)

# Get all checkpoint files
checkpoint_files = [f for f in os.listdir(CHECKPOINT_DIR) if f.endswith('.pth')]
checkpoint_files.sort(key=lambda x: os.path.getmtime(os.path.join(CHECKPOINT_DIR, x)), reverse=True)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

for ckpt_file in checkpoint_files:
    ckpt_path = os.path.join(CHECKPOINT_DIR, ckpt_file)
    try:
        checkpoint = torch.load(ckpt_path, map_location=DEVICE, weights_only=False)
        print(f"\n{ckpt_file}")
        print("-" * 50)
        if 'epoch' in checkpoint:
            print(f"Epoch: {checkpoint['epoch']}")
        if 'best_dice' in checkpoint:
            print(f"Best Dice Score: {checkpoint['best_dice']:.4f}")
        if 'best_iou' in checkpoint:
            print(f"Best IoU Score: {checkpoint['best_iou']:.4f}")
        if 'val_acc' in checkpoint:
            print(f"Val Accuracy: {checkpoint['val_acc']:.4f}")
        # Print all keys for debugging
        print(f"Checkpoint keys: {list(checkpoint.keys())}")
    except Exception as e:
        print(f"\n{ckpt_file}")
        print("-" * 50)
        print(f"Error loading checkpoint: {e}")

print("\n" + "=" * 80)

