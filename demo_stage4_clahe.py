
"""
Stage 4 Demo: LAB Color Space + CLAHE Enhancement
Visualizes the LAB conversion and CLAHE process in detail
"""
import os
import torch
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import cv2
import albumentations as A
from albumentations.pytorch import ToTensorV2
from adenseunet_simple import ADenseUNet
from preprocessing import complete_preprocessing_pipeline, lab_clahe_enhancement

# Configuration
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
IMAGE_SIZE = 224
CHECKPOINT_PATH = "checkpoints/best_adenseunet.pth"
DATA_ROOT = "archive (1)/PNG"
OUTPUT_DIR = "stage4_clahe_demo"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Load model
print("Loading segmentation model...")
model = ADenseUNet(in_channels=3, out_channels=1).to(DEVICE)
checkpoint = torch.load(CHECKPOINT_PATH, map_location=DEVICE)
model.load_state_dict(checkpoint["model_state_dict"])
model.eval()

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
        
        # Complete pipeline
        results = complete_preprocessing_pipeline(original_img, model, DEVICE)
        
        # LAB conversion details
        lab = cv2.cvtColor(results["blended_rgb"], cv2.COLOR_RGB2LAB)
        l, a, b = cv2.split(lab)
        
        # Visualize all details
        fig, axes = plt.subplots(2, 4, figsize=(20, 10))
        
        axes[0, 0].imshow(results["blended_rgb"])
        axes[0, 0].set_title("Input (Blended RGB)", fontsize=12)
        axes[0, 0].axis("off")
        
        axes[0, 1].imshow(l, cmap="gray")
        axes[0, 1].set_title("LAB: L Channel", fontsize=12)
        axes[0, 1].axis("off")
        
        axes[0, 2].imshow(a, cmap="gray")
        axes[0, 2].set_title("LAB: A Channel", fontsize=12)
        axes[0, 2].axis("off")
        
        axes[0, 3].imshow(b, cmap="gray")
        axes[0, 3].set_title("LAB: B Channel", fontsize=12)
        axes[0, 3].axis("off")
        
        # Histograms
        axes[1, 0].hist(l.flatten(), bins=256, color="black", alpha=0.7)
        axes[1, 0].set_title("L Histogram (Before)", fontsize=12)
        axes[1, 0].set_xlim([0, 255])
        
        # Apply CLAHE and show
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        l_clahe = clahe.apply(l)
        lab_clahe = cv2.merge((l_clahe, a, b))
        rgb_clahe = cv2.cvtColor(lab_clahe, cv2.COLOR_LAB2RGB)
        
        axes[1, 1].hist(l_clahe.flatten(), bins=256, color="blue", alpha=0.7)
        axes[1, 1].set_title("L Histogram (After CLAHE)", fontsize=12)
        axes[1, 1].set_xlim([0, 255])
        
        axes[1, 2].imshow(l_clahe, cmap="gray")
        axes[1, 2].set_title("CLAHE Enhanced L", fontsize=12)
        axes[1, 2].axis("off")
        
        axes[1, 3].imshow(results["after_clahe"])
        axes[1, 3].set_title("Final RGB Output", fontsize=12)
        axes[1, 3].axis("off")
        
        plt.tight_layout()
        save_path = os.path.join(OUTPUT_DIR, f"stage4_demo_{i+1}.png")
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"Saved {save_path}")

print("\nStage 4 Demo Complete!")
print(f"Check {OUTPUT_DIR}/ for LAB + CLAHE visualization!")
