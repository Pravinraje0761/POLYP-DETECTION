
import torch
import os

CHECKPOINT_DIR = "checkpoints"
checkpoint_files = [f for f in os.listdir(CHECKPOINT_DIR) if f.endswith(".pth")]

best_overall_dice = 0.0
best_checkpoint = None

print("Checking all checkpoints...")
print("=" * 80)

for ckpt_file in checkpoint_files:
    try:
        ckpt_path = os.path.join(CHECKPOINT_DIR, ckpt_file)
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        dice = ckpt.get("best_dice", 0.0)
        epoch = ckpt.get("epoch", "N/A")
        print(f"{ckpt_file:40} | Epoch: {epoch:5} | Best Dice: {dice:.4f}")
        
        if dice > best_overall_dice:
            best_overall_dice = dice
            best_checkpoint = ckpt_file
    except Exception as e:
        print(f"{ckpt_file:40} | Error loading: {e}")

print("=" * 80)
if best_checkpoint:
    print(f"\nBEST CHECKPOINT: {best_checkpoint}")
    print(f"BEST DICE SCORE: {best_overall_dice:.4f} ({best_overall_dice*100:.2f}%)")

