
"""
Stage 3 Demo: Selective RGB Enhancement
Visualize each step of the RGB enhancement process
"""
import os
import torch
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import albumentations as A
from albumentations.pytorch import ToTensorV2
from adenseunet_simple import ADenseUNet
from polypgate.preprocessing.pipeline import (
    soft_roi_extraction,
    contrast_stretching,
    gamma_correction,
    white_balance_correction,
    rgb_channel_normalization,
    selective_rgb_enhancement
)

# Configuration
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
IMAGE_SIZE = 224
CHECKPOINT_PATH = "checkpoints/best_adenseunet.pth"
DATA_ROOT = "archive (1)/PNG"
OUTPUT_DIR = "stage3_rgb_demo"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Load model
print("Loading segmentation model...")
model = ADenseUNet(in_channels=3, out_channels=1).to(DEVICE)
checkpoint = torch.load(CHECKPOINT_PATH, map_location=DEVICE)
model.load_state_dict(checkpoint["model_state_dict"])
model.eval()

# Image transforms
transform = A.Compose([
    A.Resize(IMAGE_SIZE, IMAGE_SIZE),
    A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ToTensorV2()
])

# Get 10 sample images
original_dir = os.path.join(DATA_ROOT, "Original")
sample_files = sorted(os.listdir(original_dir))[:10]
print(f"Processing {len(sample_files)} sample images...")

# Process images
with torch.no_grad():
    for i, img_file in enumerate(sample_files):
        # Load image
        img_path = os.path.join(original_dir, img_file)
        original_img = np.array(Image.open(img_path).convert("RGB"))
        
        # Get segmentation mask
        augmented = transform(image=original_img)
        img_tensor = augmented["image"].unsqueeze(0).to(DEVICE)
        outputs = model(img_tensor)
        prob_mask = torch.sigmoid(outputs).squeeze().cpu().numpy()
        prob_mask = np.array(Image.fromarray(prob_mask).resize(original_img.shape[1::-1], Image.Resampling.BILINEAR))
        
        # Apply each enhancement step by step
        img_cs = contrast_stretching(original_img)
        img_gamma = gamma_correction(img_cs, gamma=1.2)
        img_wb = white_balance_correction(img_gamma)
        img_norm = rgb_channel_normalization(img_wb)
        
        # Stage 3: Selective RGB Enhancement
        orig, enhanced, blended = selective_rgb_enhancement(original_img, prob_mask)
        
        # Visualize
        fig, axes = plt.subplots(2, 4, figsize=(20, 10))
        
        # Row 1: Enhancement steps
        axes[0, 0].imshow(original_img)
        axes[0, 0].set_title("1. Original", fontsize=12)
        axes[0, 0].axis("off")
        
        axes[0, 1].imshow(img_cs)
        axes[0, 1].set_title("2. Contrast Stretch", fontsize=12)
        axes[0, 1].axis("off")
        
        axes[0, 2].imshow(img_gamma)
        axes[0, 2].set_title("3. Gamma Correction", fontsize=12)
        axes[0, 2].axis("off")
        
        axes[0, 3].imshow(img_wb)
        axes[0, 3].set_title("4. White Balance", fontsize=12)
        axes[0, 3].axis("off")
        
        # Row 2: Final results
        axes[1, 0].imshow(img_norm)
        axes[1, 0].set_title("5. Channel Normalization", fontsize=12)
        axes[1, 0].axis("off")
        
        axes[1, 1].imshow(prob_mask, cmap="jet")
        axes[1, 1].set_title("6. Probability Mask", fontsize=12)
        axes[1, 1].axis("off")
        
        axes[1, 2].imshow(blended)
        axes[1, 2].set_title("7. Selective Blended", fontsize=12)
        axes[1, 2].axis("off")
        
        axes[1, 3].imshow(blended)
        axes[1, 3].set_title("8. Final Output", fontsize=12)
        axes[1, 3].axis("off")
        
        plt.tight_layout()
        save_path = os.path.join(OUTPUT_DIR, f"stage3_demo_{i+1}.png")
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"Saved {save_path}")

print("\nStage 3 Demo Complete!")
print(f"Check {OUTPUT_DIR}/ for visualization of selective RGB enhancement!")
