
import torch

DEVICE = torch.device("cpu")
ckpt_path = "checkpoints/best_adenseunet_boosted.pth"
checkpoint = torch.load(ckpt_path, map_location=DEVICE, weights_only=False)
print("Checkpoint keys:", list(checkpoint.keys()))
print("\nModel state dict keys:")
for key in checkpoint['model_state_dict'].keys():
    print(key)

