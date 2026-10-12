
import os
import numpy as np
import torch
import cv2
from PIL import Image
import matplotlib.pyplot as plt
from tqdm import tqdm
import albumentations as A
from albumentations.pytorch import ToTensorV2

# Import our models and functions
from polypgate.models.adenseunet import ADenseUNet_Boosted
from polypgate.preprocessing.pipeline import (
    soft_roi_extraction,
    selective_rgb_enhancement,
    lab_clahe_enhancement
)

# Create output directory
OUTPUT_DIR = "hyperkvasir_all_11_stages_best_model"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Hyper-Kvasir directory
HYPER_KVASIR_DIR = r"C:\Users\ADMIN\Desktop\POLYP_DETECTION\hyper-kvasir-labeled-images (1)\labeled-images\lower-gi-tract\pathological-findings\polyps"

# Load our best model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = ADenseUNet_Boosted(in_channels=3, out_channels=1).to(device)
checkpoint = torch.load("checkpoints/best_adenseunet_boosted.pth", map_location=device, weights_only=False)
model.load_state_dict(checkpoint["model_state_dict"])
model.eval()

print("="*80)
print("Starting Hyper-Kvasir 11 Stages Analysis (ALL Images)")
print(f"Model: best_adenseunet_boosted.pth (Best Dice: {checkpoint.get('best_dice', 0):.6f})")
print("="*80)

# Get all images
image_files = sorted([f for f in os.listdir(HYPER_KVASIR_DIR) if f.lower().endswith((".jpg", ".jpeg", ".png"))])
print(f"Found {len(image_files)} images in total")

# Define transform for model input
transform = A.Compose([
    A.Resize(224, 224),
    A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ToTensorV2(),
])

# Process each image
for idx, img_file in enumerate(tqdm(image_files, desc="Processing images")):
    try:
        # Load original image
        img_path = os.path.join(HYPER_KVASIR_DIR, img_file)
        original_img = np.array(Image.open(img_path).convert("RGB"))
        
        # Stage 1: Segmentation
        img_input = transform(image=original_img)["image"].unsqueeze(0).to(device)
        with torch.no_grad():
            logits = model(img_input)
            pred_mask = torch.sigmoid(logits)[0, 0].cpu().numpy()
        pred_mask_resized = cv2.resize(pred_mask, (original_img.shape[1], original_img.shape[0]))
        
        # Stages 2-4: Preprocessing pipeline
        _, _, roi_img = soft_roi_extraction(original_img, pred_mask_resized)
        _, enhanced_rgb, blended_rgb = selective_rgb_enhancement(original_img, pred_mask_resized)
        before_clahe, after_clahe = lab_clahe_enhancement(blended_rgb)
        
        # Create visualization of all 11 stages
        fig, axes = plt.subplots(3, 4, figsize=(20, 15))
        fig.suptitle(f"All 11 Stages (Best Model): {img_file}", fontsize=16, y=0.98)
        
        # ==================
        # Stage 1: Original Image
        # ==================
        axes[0, 0].imshow(original_img)
        axes[0, 0].set_title("Stage 1: Original Image")
        axes[0, 0].axis("off")
        
        # ==================
        # Stage 2: Segmentation Mask
        # ==================
        axes[0, 1].imshow(pred_mask_resized, cmap="gray")
        axes[0, 1].set_title("Stage 2: Segmentation Mask")
        axes[0, 1].axis("off")
        
        # ==================
        # Stage 3: Segmentation Overlay
        # ==================
        axes[0, 2].imshow(original_img)
        axes[0, 2].imshow(pred_mask_resized, cmap="jet", alpha=0.4)
        axes[0, 2].set_title("Stage 3: Segmentation Overlay")
        axes[0, 2].axis("off")
        
        # ==================
        # Stage 4: ROI Extraction
        # ==================
        axes[0, 3].imshow(roi_img)
        axes[0, 3].set_title("Stage 4: ROI Extraction")
        axes[0, 3].axis("off")
        
        # ==================
        # Stage 5: RGB Enhanced
        # ==================
        axes[1, 0].imshow(enhanced_rgb)
        axes[1, 0].set_title("Stage 5: RGB Enhanced")
        axes[1, 0].axis("off")
        
        # ==================
        # Stage 6: RGB Blended
        # ==================
        axes[1, 1].imshow(blended_rgb)
        axes[1, 1].set_title("Stage 6: RGB Blended")
        axes[1, 1].axis("off")
        
        # ==================
        # Stage 7: Before CLAHE
        # ==================
        axes[1, 2].imshow(before_clahe)
        axes[1, 2].set_title("Stage 7: Before CLAHE")
        axes[1, 2].axis("off")
        
        # ==================
        # Stage 8: After CLAHE
        # ==================
        axes[1, 3].imshow(after_clahe)
        axes[1, 3].set_title("Stage 8: After CLAHE")
        axes[1, 3].axis("off")
        
        # ==================
        # Stage 9: Binary Mask
        # ==================
        binary_mask = (pred_mask_resized > 0.5).astype(np.uint8) * 255
        axes[2, 0].imshow(binary_mask, cmap="gray")
        axes[2, 0].set_title("Stage 9: Binary Mask")
        axes[2, 0].axis("off")
        
        # ==================
        # Stage 10: Contour Detection
        # ==================
        contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contour_img = original_img.copy()
        cv2.drawContours(contour_img, contours, -1, (0, 255, 0), 2)
        axes[2, 1].imshow(contour_img)
        axes[2, 1].set_title("Stage 10: Contour Detection")
        axes[2, 1].axis("off")
        
        # ==================
        # Stage 11: Mask Intensity Histogram
        # ==================
        axes[2, 2].hist(pred_mask_resized.flatten(), bins=50, range=[0, 1], color="blue", alpha=0.7)
        axes[2, 2].set_title("Stage 11: Mask Intensity Histogram")
        axes[2, 2].set_xlabel("Probability")
        axes[2, 2].set_ylabel("Frequency")
        
        # ==================
        # Summary
        # ==================
        axes[2, 3].axis("off")
        summary_text = f"""
Summary

Image: {img_file}
Mask Coverage: {np.mean(pred_mask_resized)*100:.1f}%
Max Probability: {np.max(pred_mask_resized):.2f}
Min Probability: {np.min(pred_mask_resized):.2f}

All 11 Stages Complete!
"""
        axes[2, 3].text(0.1, 0.5, summary_text, ha="left", va="center", fontsize=10, 
                       transform=axes[2, 3].transAxes, bbox=dict(boxstyle="round", facecolor="#f0f0f0"))
        
        plt.tight_layout()
        output_filename = f"stage_analysis_{idx+1:04d}_{os.path.splitext(img_file)[0]}.png"
        plt.savefig(os.path.join(OUTPUT_DIR, output_filename), dpi=150, bbox_inches="tight")
        plt.close()
        
    except Exception as e:
        print(f"\nError processing {img_file}: {str(e)}")
        continue

print("\n" + "="*80)
print(f"✅ All 11 stages complete for {len(image_files)} images!")
print(f"Results saved to: {os.path.abspath(OUTPUT_DIR)}")
print("="*80)
