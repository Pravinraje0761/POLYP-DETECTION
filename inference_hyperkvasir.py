
"""
Inference script for Stage1: A-DenseUNet on Hyper-Kvasir polyp images
"""
import os
import torch
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import albumentations as A
from albumentations.pytorch import ToTensorV2
from adenseunet_simple import ADenseUNet

# Configuration
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
IMAGE_SIZE = 224
CHECKPOINT_PATH = "checkpoints/best_adenseunet.pth"
HYPER_KVASIR_POLYP_DIR = r"c:\Users\ADMIN\Desktop\POLYP_DETECTION\hyper-kvasir-labeled-images (1)\labeled-images\lower-gi-tract\pathological-findings\polyps"
OUTPUT_DIR = "hyperkvasir_predictions"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Load model
print("Loading model...")
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

# Get sample images (next 10, to avoid repeating previous ones)
sample_files = sorted(os.listdir(HYPER_KVASIR_POLYP_DIR))[10:20]
print(f"Processing {len(sample_files)} new sample images from Hyper-Kvasir...")

# Inference
with torch.no_grad():
    for i, img_file in enumerate(sample_files):
        # Load image
        img_path = os.path.join(HYPER_KVASIR_POLYP_DIR, img_file)
        original_img = np.array(Image.open(img_path).convert("RGB"))
        
        # Prepare tensor
        augmented = transform(image=original_img)
        img_tensor = augmented["image"].unsqueeze(0).to(DEVICE)
        
        # Predict
        outputs = model(img_tensor)
        pred_mask = torch.sigmoid(outputs).squeeze(0).squeeze(0).cpu().numpy()
        pred_mask_thresh = (pred_mask > 0.5).astype(np.float32)
        
        # Resize back to original size
        pred_mask_resized = np.array(Image.fromarray(pred_mask_thresh).resize(original_img.shape[1::-1], Image.Resampling.BILINEAR))
        
        # Plot and save
        fig, axes = plt.subplots(1, 2, figsize=(12, 6))
        axes[0].imshow(original_img)
        axes[0].set_title("Original Image (Hyper-Kvasir)")
        axes[0].axis("off")
        
        # Overlay mask on original image
        overlay = original_img.copy()
        overlay[pred_mask_resized > 0.5] = [255, 0, 0]  # Red overlay
        axes[1].imshow(overlay)
        axes[1].set_title("Predicted Polyp Mask")
        axes[1].axis("off")
        
        save_path = os.path.join(OUTPUT_DIR, f"hk_polyp_{i+11}.png")
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"Saved sample {i+1} to {save_path}")

print("Done! All Hyper-Kvasir predictions are in hyperkvasir_predictions/ directory!")
