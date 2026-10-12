
import torch
import torch.nn as nn

# Let's load the checkpoint and see the exact shapes
checkpoint = torch.load("checkpoints/best_adenseunet_boosted.pth", map_location="cpu", weights_only=False)
state_dict = checkpoint["model_state_dict"]

print("State dict keys and shapes:")
for k, v in state_dict.items():
    print(f"{k}: {v.shape}")

print("\nLooking for decoder keys:")
print("final_up or up_final?", [k for k in state_dict.keys() if "final" in k or "up" in k])
