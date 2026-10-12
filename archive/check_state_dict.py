
import torch

checkpoint = torch.load("checkpoints/best_adenseunet_boosted.pth", map_location="cpu", weights_only=False)
print("Checkpoint keys available:", checkpoint.keys())
print("\nModel state dict keys:")
for i, key in enumerate(checkpoint["model_state_dict"].keys()):
    print(f"{i+1}. {key}")
