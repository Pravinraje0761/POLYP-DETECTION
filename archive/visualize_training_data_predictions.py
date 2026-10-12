
"""
Visualize predictions on the TRAINING dataset (archive (1)/PNG)
with ground truth masks for comparison
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
OUTPUT_DIR = "training_data_predictions"
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

# Get 10 sample images
original_dir = os.path.join(DATA_ROOT, "Original")
gt_dir = os.path.join(DATA_ROOT, "Ground Truth")
sample_files = sorted(os.listdir(original_dir))[:10]
print(f"Processing {len(sample_files)} sample images from training dataset...")

# Inference and visualization
with torch.no_grad():
    for i, img_file in enumerate(sample_files):
        # Load image and ground truth mask
        img_path = os.path.join(original_dir, img_file)
        gt_path = os.path.join(gt_dir, img_file)
        
        original_img = np.array(Image.open(img_path).convert("RGB"))
        original_gt = np.array(Image.open(gt_path).convert("L")) / 255.0
        
        # Prepare tensor
        augmented = transform(image=original_img)
        img_tensor = augmented["image"].unsqueeze(0).to(DEVICE)
        
        # Predict
        outputs = model(img_tensor)
        pred_prob = torch.sigmoid(outputs).squeeze(0).squeeze(0).cpu().numpy()
        
        # Resize back to original size
        pred_prob_resized = np.array(Image.fromarray(pred_prob).resize(
            original_img.shape[1::-1], Image.Resampling.BILINEAR
        ))
        pred_mask = (pred_prob_resized > 0.5).astype(np.float32)
        
        # Plot 4 panels: original, GT mask, raw probability, thresholded mask
        fig, axes = plt.subplots(1, 4, figsize=(20, 5))
        
        axes[0].imshow(original_img)
        axes[0].set_title("Original Image")
        axes[0].axis("off")
        
        axes[1].imshow(original_gt, cmap="gray")
        axes[1].set_title("Ground Truth Mask")
        axes[1].axis("off")
        
        axes[2].imshow(pred_prob_resized, cmap="jet", vmin=0, vmax=1)
        axes[2].set_title("Raw Prediction Probability")
        axes[2].axis("off")
        
        # Overlay thresholded mask on original image
        overlay = original_img.copy()
        overlay[pred_mask > 0.5] = [255, 0, 0]  # Red overlay
        axes[3].imshow(overlay)
        axes[3].set_title("Thresholded Prediction Overlay")
        axes[3].axis("off")
        
        plt.tight_layout()
        save_path = os.path.join(OUTPUT_DIR, f"train_pred_{i+1}.png")
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"Saved {save_path}")

print("\nDone! Check training_data_predictions/ for results!")
print("\nNote: The model was trained on this dataset, so predictions should be accurate here!")
