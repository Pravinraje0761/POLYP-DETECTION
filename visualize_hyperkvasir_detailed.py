
"""
Detailed visualization for Hyper-Kvasir with raw probabilities
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
OUTPUT_DIR = "hyperkvasir_detailed_predictions"
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
sample_files = sorted(os.listdir(HYPER_KVASIR_POLYP_DIR))[:10]
print(f"Processing {len(sample_files)} sample images from Hyper-Kvasir...")

# Inference and visualization
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
        pred_prob = torch.sigmoid(outputs).squeeze(0).squeeze(0).cpu().numpy()
        
        # Resize back to original size
        pred_prob_resized = np.array(Image.fromarray(pred_prob).resize(
            original_img.shape[1::-1], Image.Resampling.BILINEAR
        ))
        
        # Try different thresholds
        pred_mask_03 = (pred_prob_resized > 0.3).astype(np.float32)
        pred_mask_05 = (pred_prob_resized > 0.5).astype(np.float32)
        pred_mask_07 = (pred_prob_resized > 0.7).astype(np.float32)
        
        # Plot detailed visualization
        fig, axes = plt.subplots(1, 5, figsize=(25, 5))
        
        axes[0].imshow(original_img)
        axes[0].set_title("Original Image")
        axes[0].axis("off")
        
        axes[1].imshow(pred_prob_resized, cmap="jet", vmin=0, vmax=1)
        axes[1].set_title("Raw Prediction Probability")
        axes[1].axis("off")
        
        # Overlays with different thresholds
        overlay_03 = original_img.copy()
        overlay_03[pred_mask_03 > 0.5] = [255, 0, 0]
        axes[2].imshow(overlay_03)
        axes[2].set_title("Threshold: 0.3")
        axes[2].axis("off")
        
        overlay_05 = original_img.copy()
        overlay_05[pred_mask_05 > 0.5] = [255, 0, 0]
        axes[3].imshow(overlay_05)
        axes[3].set_title("Threshold: 0.5")
        axes[3].axis("off")
        
        overlay_07 = original_img.copy()
        overlay_07[pred_mask_07 > 0.5] = [255, 0, 0]
        axes[4].imshow(overlay_07)
        axes[4].set_title("Threshold: 0.7")
        axes[4].axis("off")
        
        plt.tight_layout()
        save_path = os.path.join(OUTPUT_DIR, f"hk_detailed_{i+1}.png")
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"Saved {save_path}")

print("\nDone! Check hyperkvasir_detailed_predictions/ for detailed results!")
print("\nImportant note: The model was trained on a different dataset (archive (1)/PNG),")
print("so performance on Hyper-Kvasir might vary. For best results, fine-tune on Hyper-Kvasir!")
