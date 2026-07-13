
import torch
from adenseunet_boosted import ADenseUNet_Boosted
ckpt_path = "checkpoints/best_adenseunet_boosted.pth"
checkpoint = torch.load(ckpt_path, map_location="cpu", weights_only=False)
state_dict = checkpoint["model_state_dict"]
model = ADenseUNet_Boosted(in_channels=3, out_channels=1)
model_state_dict = model.state_dict()

print("Checkpoint layers:")
for key in list(state_dict.keys()):
    print(key)

print("\nModel layers:")
for key in list(model_state_dict.keys()):
    print(key)
