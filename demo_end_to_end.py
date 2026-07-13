
"""
End-to-End Pipeline Demo: Stages 1-4
Visualizes the complete preprocessing pipeline
"""
import os
import torch
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import albumentations as A
from albumentations.pytorch import ToTensorV2
from adenseunet_simple import ADenseUNet
from preprocessing import complete_preprocessing_pipeline

# Configuration
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
IMAGE_SIZE = 224
CHECKPOINT_PATH = "checkpoints/best_adenseunet.pth"
DATA_ROOT = "archive (1)/PNG"
OUTPUT_DIR = "end_to_end_demo"
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
        
        # Visualize all stages
        fig, axes = plt.subplots(2, 4, figsize=(20, 10))
        
        axes[0, 0].imshow(results["original"])
        axes[0, 0].set_title("Stage 1: Input", fontsize=12)
        axes[0, 0].axis("off")
        
        im = axes[0, 1].imshow(results["prob_mask"], cmap="jet")
        axes[0, 1].set_title("Stage 1: Segmentation Mask", fontsize=12)
        axes[0, 1].axis("off")
        plt.colorbar(im, ax=axes[0, 1], fraction=0.046, pad=0.04)
        
        axes[0, 2].imshow(results["roi"])
        axes[0, 2].set_title("Stage 2: Soft ROI", fontsize=12)
        axes[0, 2].axis("off")
        
        axes[0, 3].imshow(results["enhanced_rgb"])
        axes[0, 3].set_title("Stage 3: Enhanced (Full)", fontsize=12)
        axes[0, 3].axis("off")
        
        axes[1, 0].imshow(results["blended_rgb"])
        axes[1, 0].set_title("Stage 3: Enhanced (Blended)", fontsize=12)
        axes[1, 0].axis("off")
        
        axes[1, 1].imshow(results["before_clahe"])
        axes[1, 1].set_title("Stage 4: Before CLAHE", fontsize=12)
        axes[1, 1].axis("off")
        
        axes[1, 2].imshow(results["after_clahe"])
        axes[1, 2].set_title("Stage 4: After CLAHE", fontsize=12)
        axes[1, 2].axis("off")
        
        axes[1, 3].imshow(results["after_clahe"])
        axes[1, 3].set_title("Final Output", fontsize=12)
        axes[1, 3].axis("off")
        
        plt.tight_layout()
        save_path = os.path.join(OUTPUT_DIR, f"e2e_demo_{i+1}.png")
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"Saved {save_path}")

print("\nEnd-to-End Pipeline Demo Complete!")
print(f"Check {OUTPUT_DIR}/ for visualization of the complete pipeline!")
