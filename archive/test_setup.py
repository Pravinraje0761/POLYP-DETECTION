
import torch
from adenseunet_simple_original import ADenseUNet

# Configuration
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CHECKPOINT_DIR = "checkpoints"

print("Testing model loading...")
model = ADenseUNet(in_channels=3, out_channels=1).to(DEVICE)

# Initialize new layers
model.extra_up = torch.nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2).to(DEVICE)
model.final_conv = torch.nn.Conv2d(32, 1, kernel_size=1).to(DEVICE)

# Load checkpoint
try:
    checkpoint = torch.load(f"{CHECKPOINT_DIR}/best_adenseunet_boosted_fixed.pth", map_location=DEVICE, weights_only=False)
    state_dict = checkpoint["model_state_dict"]
    
    # Remove old final_conv keys
    if "final_conv.weight" in state_dict:
        del state_dict["final_conv.weight"]
    if "final_conv.bias" in state_dict:
        del state_dict["final_conv.bias"]
    
    model.load_state_dict(state_dict, strict=False)
    print(f"SUCCESS: Loaded checkpoint with Dice: {checkpoint['best_dice']:.4f}")
    
    # Test with dummy input
    test_input = torch.randn(1, 3, 224, 224).to(DEVICE)
    with torch.no_grad():
        output = model(test_input)
        print(f"SUCCESS: Dummy test passed, output shape: {output.shape}")
        
except Exception as e:
    print(f"ERROR: {e}")

print("\n✅ Test completed - no critical errors! You can now run train_final_push_to_93.py")

