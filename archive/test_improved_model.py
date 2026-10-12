
import torch
from adenseunet_simple import ADenseUNet

print("Testing adenseunet_simple with best_adenseunet_improved.pth...")
model = ADenseUNet(in_channels=3, out_channels=1)
checkpoint = torch.load("checkpoints/best_adenseunet_improved.pth", map_location="cpu", weights_only=False)
try:
    model.load_state_dict(checkpoint["model_state_dict"])
    print("OK: Model loads successfully!")
    print(f"   Best Dice: {checkpoint.get('best_dice', 'N/A'):.6f}")
except Exception as e:
    print(f"ERROR: {e}")
