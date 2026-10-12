
"""
Inference script for Stage1: A-DenseUNet
Loads the best checkpoint and generates predictions on 5 sample images
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
DATA_ROOT = "archive (1)/PNG"
OUTPUT_DIR = "stage1_inference"
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

# Get sample images
original_dir = os.path.join(DATA_ROOT, "Original")
gt_dir = os.path.join(DATA_ROOT, "Ground Truth")
sample_files = sorted(os.listdir(original_dir))[:5]
print(f"Processing {len(sample_files)} sample images...")

# Inference
with torch.no_grad():
    for i, img_file in enumerate(sample_files):
        # Load image and mask
        img_path = os.path.join(original_dir, img_file)
        gt_path = os.path.join(gt_dir, img_file)
        original_img = np.array(Image.open(img_path).convert("RGB"))
        original_gt = np.array(Image.open(gt_path).convert("L"))
        
        # Prepare tensor
        augmented = transform(image=original_img)
        img_tensor = augmented["image"].unsqueeze(0).to(DEVICE)
        
        # Predict
        outputs = model(img_tensor)
        pred_mask = torch.sigmoid(outputs).squeeze(0).squeeze(0).cpu().numpy()
        pred_mask = (pred_mask > 0.5).astype(np.float32)  # Threshold
        
        # Resize back to original size
        pred_mask_resized = np.array(Image.fromarray(pred_mask).resize(original_img.shape[1::-1], Image.Resampling.BILINEAR))
        
        # Plot and save
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        axes[0].imshow(original_img)
        axes[0].set_title("Original Image")
        axes[0].axis("off")
        
        axes[1].imshow(original_gt, cmap="gray")
        axes[1].set_title("Ground Truth Mask")
        axes[1].axis("off")
        
        axes[2].imshow(pred_mask_resized, cmap="gray")
        axes[2].set_title("Predicted Mask")
        axes[2].axis("off")
        
        save_path = os.path.join(OUTPUT_DIR, f"sample_{i+1}.png")
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"Saved sample {i+1} to {save_path}")

print("Done! All sample outputs are in stage1_inference/ directory!")
