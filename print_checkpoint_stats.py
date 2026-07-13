
import torch
import os

checkpoint_dir = "checkpoints"
checkpoint_files = [f for f in os.listdir(checkpoint_dir) if f.endswith(".pth")]

print("=== Checkpoint Best Dice Scores ===")
all_dice = []

for ckpt_file in checkpoint_files:
    ckpt_path = os.path.join(checkpoint_dir, ckpt_file)
    try:
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        # Get dice
        dice = None
        if "best_dice" in ckpt:
            dice = ckpt["best_dice"]
        elif "val_dice" in ckpt:
            dice = ckpt["val_dice"]
        elif "dice" in ckpt:
            dice = ckpt["dice"]
        
        if dice is not None:
            print(f"{ckpt_file:<40} Dice: {dice:.6f}")
            all_dice.append((dice, ckpt_file))
        else:
            print(f"{ckpt_file:<40} (No dice metric found)")
    except Exception as e:
        print(f"{ckpt_file:<40} ERROR: {str(e)[:50]}")

print("\n=== Top Checkpoints ===")
all_dice.sort(reverse=True, key=lambda x: x[0])
for i, (dice, filename) in enumerate(all_dice[:5], start=1):
    print(f"{i}. {filename:<40} Dice: {dice:.6f}")
