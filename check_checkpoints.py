
import torch
import os

checkpoint_dir = "checkpoints"
checkpoint_files = [f for f in os.listdir(checkpoint_dir) if f.endswith(".pth")]

print("=== Checkpoint Details ===")
best_dice = 0
best_checkpoint = None

for ckpt_file in checkpoint_files:
    ckpt_path = os.path.join(checkpoint_dir, ckpt_file)
    try:
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        print(f"\n{ckpt_file}:")
        for key in ckpt:
            if key != "model_state_dict":
                print(f"  {key}: {ckpt[key]}")
        
        # Check for dice-related metrics
        dice_score = None
        if "best_dice" in ckpt:
            dice_score = ckpt["best_dice"]
        elif "val_dice" in ckpt:
            dice_score = ckpt["val_dice"]
        
        if dice_score and dice_score > best_dice:
            best_dice = dice_score
            best_checkpoint = ckpt_file
    except Exception as e:
        print(f"\n{ckpt_file}: Error loading - {e}")

print(f"\n=== Best Checkpoint ===")
print(f"File: {best_checkpoint}")
print(f"Best Dice: {best_dice:.4f}")
