
import torch

def check_checkpoint(path, name):
    try:
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
        print(f"\n{name}")
        print("=" * 50)
        print(f"Epoch: {checkpoint.get('epoch', 'N/A')}")
        print(f"Best Dice: {checkpoint.get('best_dice', 'N/A'):.4f}")
    except Exception as e:
        print(f"\n{name}: {e}")

check_checkpoint("checkpoints/best_adenseunet.pth", "Original Checkpoint")
check_checkpoint("checkpoints/best_adenseunet_improved.pth", "Improved Checkpoint")
check_checkpoint("checkpoints/best_adenseunet_ultimate.pth", "Ultimate Checkpoint")
