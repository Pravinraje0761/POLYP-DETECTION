
import torch

# Check improved segmentation checkpoint
checkpoint_path = "checkpoints/best_adenseunet_improved.pth"
checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
print("=" * 80)
print("Improved Checkpoint Info:")
for key in checkpoint.keys():
    if isinstance(checkpoint[key], (int, float, str)):
        print(f"  {key}: {checkpoint[key]}")
print("=" * 80)

# Check original segmentation checkpoint
checkpoint_path2 = "checkpoints/best_adenseunet.pth"
checkpoint2 = torch.load(checkpoint_path2, map_location="cpu", weights_only=False)
print("\nOriginal Checkpoint Info:")
for key in checkpoint2.keys():
    if isinstance(checkpoint2[key], (int, float, str)):
        print(f"  {key}: {checkpoint2[key]}")
print("=" * 80)

# Check ultimate segmentation checkpoint
checkpoint_path3 = "checkpoints/best_adenseunet_ultimate.pth"
checkpoint3 = torch.load(checkpoint_path3, map_location="cpu", weights_only=False)
print("\nUltimate Checkpoint Info:")
for key in checkpoint3.keys():
    if isinstance(checkpoint3[key], (int, float, str)):
        print(f"  {key}: {checkpoint3[key]}")
print("=" * 80)
