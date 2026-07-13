
import torch

checkpoints = [
    ("Original", "checkpoints/best_adenseunet.pth"),
    ("Improved", "checkpoints/best_adenseunet_improved.pth"),
    ("Boosted", "checkpoints/best_adenseunet_boosted.pth"),
    ("Boosted Fixed", "checkpoints/best_adenseunet_boosted_fixed.pth"),
    ("93 Dice", "checkpoints/best_adenseunet_93_dice.pth"),
    ("Ultimate", "checkpoints/best_adenseunet_ultimate.pth"),
    ("Ultimate Fixed", "checkpoints/best_adenseunet_ultimate_fixed.pth")
]

print("=" * 70)
print("POLYP SEGMENTATION MODEL CHECKPOINTS")
print("=" * 70)

for name, path in checkpoints:
    try:
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
        print(f"\n{name:15}")
        print("-" * 40)
        print(f"Epoch: {checkpoint.get('epoch', 'N/A')}")
        print(f"Best Dice: {checkpoint.get('best_dice', 'N/A'):.4f}")
    except Exception as e:
        print(f"\n{name:15}")
        print(f"Error: {str(e)}")

print("\n" + "=" * 70)
