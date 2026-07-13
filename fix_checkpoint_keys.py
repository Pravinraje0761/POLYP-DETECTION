
"""
Fix checkpoint keys for modified ADenseUNet (rename final_up → up_final)
"""
import torch
import os

CHECKPOINT_DIR = "checkpoints"
OLD_CHECKPOINT = os.path.join(CHECKPOINT_DIR, "best_adenseunet_ultimate.pth")
NEW_CHECKPOINT = os.path.join(CHECKPOINT_DIR, "best_adenseunet_ultimate_fixed.pth")

if os.path.exists(OLD_CHECKPOINT):
    checkpoint = torch.load(OLD_CHECKPOINT, map_location='cpu', weights_only=False)
    state_dict = checkpoint["model_state_dict"]
    
    # Rename keys
    new_state_dict = {}
    for key, value in state_dict.items():
        if key.startswith("final_up"):
            new_key = key.replace("final_up", "up_final")
            new_state_dict[new_key] = value
        else:
            new_state_dict[key] = value
    
    checkpoint["model_state_dict"] = new_state_dict
    torch.save(checkpoint, NEW_CHECKPOINT)
    print(f"Fixed checkpoint saved to {NEW_CHECKPOINT}")
else:
    print(f"Checkpoint {OLD_CHECKPOINT} not found!")

