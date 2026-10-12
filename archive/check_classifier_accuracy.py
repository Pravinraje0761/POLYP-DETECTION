
"""
Script to check the ViT classifier accuracy
"""
import torch

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CKPT_PATH = "outputs/checkpoints/best_model.pth"

print("=" * 80)
print("VIT CLASSIFIER ACCURACY")
print("=" * 80)

try:
    checkpoint = torch.load(CKPT_PATH, map_location=DEVICE, weights_only=False)
    print(f"\nCheckpoint Path: {CKPT_PATH}")
    print("-" * 50)
    for key, value in checkpoint.items():
        if isinstance(value, (int, float)):
            print(f"{key}: {value}")
        elif key not in ['model_state_dict', 'optimizer_state_dict', 'scheduler_state_dict']:
            print(f"{key}: {value}")
except Exception as e:
    print(f"Error loading checkpoint: {e}")

print("\n" + "=" * 80)
