
import torch
from adenseunet_boosted import ADenseUNet_Boosted

print("Testing best_adenseunet_boosted.pth...")
model1 = ADenseUNet_Boosted(in_channels=3, out_channels=1)
ckpt1 = torch.load("checkpoints/best_adenseunet_boosted.pth", map_location="cpu", weights_only=False)
try:
    model1.load_state_dict(ckpt1["model_state_dict"])
    print("OK: best_adenseunet_boosted.pth loads successfully!")
    print(f"   Best Dice: {ckpt1['best_dice']:.6f}")
except Exception as e:
    print(f"ERROR: best_adenseunet_boosted.pth failed: {e}")

print("\nTesting best_adenseunet_boosted_fixed.pth...")
model2 = ADenseUNet_Boosted(in_channels=3, out_channels=1)
ckpt2 = torch.load("checkpoints/best_adenseunet_boosted_fixed.pth", map_location="cpu", weights_only=False)
try:
    model2.load_state_dict(ckpt2["model_state_dict"])
    print("OK: best_adenseunet_boosted_fixed.pth loads successfully!")
    print(f"   Best Dice: {ckpt2['best_dice']:.6f}")
except Exception as e:
    print(f"ERROR: best_adenseunet_boosted_fixed.pth failed: {e}")
