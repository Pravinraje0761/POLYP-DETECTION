
"""
Check the improved segmentation checkpoint
"""
import torch

checkpoint_path = "checkpoints/best_adenseunet_improved.pth"

try:
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    print("=" * 50)
    print("Improved Segmentation Checkpoint Info")
    print("=" * 50)
    print(f"Epoch trained: {checkpoint.get('epoch', 'N/A')}")
    print(f"Best validation Dice: {checkpoint.get('best_dice', 'N/A'):.4f}")
    print("=" * 50)
except FileNotFoundError:
    print("Improved checkpoint not found, checking original...")
    checkpoint = torch.load("checkpoints/best_adenseunet.pth", map_location="cpu", weights_only=False)
    print("=" * 50)
    print("Original Segmentation Checkpoint Info")
    print("=" * 50)
    print(f"Epoch trained: {checkpoint.get('epoch', 'N/A')}")
    print(f"Best validation Dice: {checkpoint.get('best_dice', 'N/A'):.4f}")
    print("=" * 50)
