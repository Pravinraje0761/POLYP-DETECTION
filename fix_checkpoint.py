
import torch

DEVICE = torch.device("cpu")
ckpt_path = "checkpoints/best_adenseunet_boosted.pth"
new_ckpt_path = "checkpoints/best_adenseunet_boosted_fixed.pth"

checkpoint = torch.load(ckpt_path, map_location=DEVICE, weights_only=False)
state_dict = checkpoint['model_state_dict']

new_state_dict = {}
for key, value in state_dict.items():
    if key == "up_final.weight":
        new_key = "final_up.weight"
    elif key == "up_final.bias":
        new_key = "final_up.bias"
    else:
        new_key = key
    new_state_dict[new_key] = value

checkpoint['model_state_dict'] = new_state_dict
torch.save(checkpoint, new_ckpt_path)
print(f"Fixed checkpoint saved to {new_ckpt_path}")
print("Renamed keys: up_final -> final_up")

