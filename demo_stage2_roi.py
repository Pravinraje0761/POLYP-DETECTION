
"""
Stage 2 Demo: Soft ROI Extraction
Visualize the ROI extraction process on sample images
"""
import os
import torch
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import albumentations as A
from albumentations.pytorch import ToTensorV2
from adenseunet_simple import ADenseUNet
from preprocessing import soft_roi_extraction

# Configuration
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
IMAGE_SIZE = 224
CHECKPOINT_PATH = "checkpoints/best_adenseunet.pth"
DATA_ROOT = "archive (1)/PNG"
OUTPUT_DIR = "stage2_roi_demo"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Load model
print("Loading segmentation model...")
model = ADenseUNet(in_channels=3, out_channels=1).to(DEVICE)
checkpoint = torch.load(CHECKPOINT_PATH, map_location=DEVICE)
model.load_state_dict(checkpoint["model_state_dict"])
model.eval()

# Image transforms for segmentation
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
        
        # Stage 2: Soft ROI Extraction
        orig, mask, roi = soft_roi_extraction(original_img, prob_mask)
        
        # Visualize
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        
        axes[0].imshow(orig)
        axes[0].set_title("Original Image", fontsize=14)
        axes[0].axis("off")
        
        im = axes[1].imshow(mask, cmap="jet", vmin=0, vmax=1)
        axes[1].set_title("Probability Mask", fontsize=14)
        axes[1].axis("off")
        plt.colorbar(im, ax=axes[1], fraction=0.046, pad=0.04)
        
        axes[2].imshow(roi)
        axes[2].set_title("Soft ROI (Image × Mask)", fontsize=14)
        axes[2].axis("off")
        
        plt.tight_layout()
        save_path = os.path.join(OUTPUT_DIR, f"stage2_demo_{i+1}.png")
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"Saved {save_path}")

print("\n✅ Stage 2 Demo Complete!")
print(f"Check {OUTPUT_DIR}/ for visualization of soft ROI extraction!")
