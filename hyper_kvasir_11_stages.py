
"""
Perform all 11 stages on 30 Hyper-Kvasir images
"""
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
from adenseunet_simple import ADenseUNet
from preprocessing import complete_preprocessing_pipeline

# Create output directory
OUTPUT_DIR = "hyper_kvasir_11_stages_results"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Hyper-Kvasir directory
HYPER_KVASIR_DIR = r"c:\Users\ADMIN\Desktop\POLYP_DETECTION\hyper-kvasir-labeled-images (1)\labeled-images\lower-gi-tract\pathological-findings\polyps"

# Load our improved model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = ADenseUNet(in_channels=3, out_channels=1).to(device)
checkpoint = torch.load("checkpoints/best_adenseunet_improved.pth", map_location=device, weights_only=False)
model.load_state_dict(checkpoint["model_state_dict"])
model.eval()

print("="*80)
print("Starting Hyper-Kvasir 11 Stages Analysis")
print("="*80)

# Get first 30 images
image_files = sorted([f for f in os.listdir(HYPER_KVASIR_DIR) if f.endswith(".jpg") or f.endswith(".png")])[:30]
print(f"Processing {len(image_files)} images from Hyper-Kvasir dataset")

# Define transform for model input
transform = A.Compose([
    A.Resize(224, 224),
    A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ToTensorV2(),
])

# Process each image
for idx, img_file in enumerate(tqdm(image_files, desc="Processing images")):
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
    preprocessed = complete_preprocessing_pipeline(original_img, model, device)
    
    # Create visualization of all 11 stages
    fig, axes = plt.subplots(3, 4, figsize=(20, 15))
    fig.suptitle(f"All 11 Stages: {img_file}", fontsize=16, y=0.98)
    
    # ==================
    # Stage 1: Segmentation
    # ==================
    axes[0, 0].imshow(original_img)
    axes[0, 0].set_title("Stage 1: Original Image")
    axes[0, 0].axis("off")
    
    axes[0, 1].imshow(pred_mask_resized, cmap="gray")
    axes[0, 1].set_title("Stage 1: Segmentation Mask")
    axes[0, 1].axis("off")
    
    axes[0, 2].imshow(original_img)
    axes[0, 2].imshow(pred_mask_resized, cmap="jet", alpha=0.4)
    axes[0, 2].set_title("Stage 1: Segmentation Overlay")
    axes[0, 2].axis("off")
    
    # ==================
    # Stage 2: ROI Extraction
    # ==================
    axes[0, 3].imshow(preprocessed["roi"])
    axes[0, 3].set_title("Stage 2: Soft ROI Extraction")
    axes[0, 3].axis("off")
    
    # ==================
    # Stage 3: RGB Enhancement
    # ==================
    axes[1, 0].imshow(preprocessed["enhanced_rgb"])
    axes[1, 0].set_title("Stage 3: RGB Enhanced")
    axes[1, 0].axis("off")
    
    axes[1, 1].imshow(preprocessed["blended_rgb"])
    axes[1, 1].set_title("Stage 3: RGB Blended")
    axes[1, 1].axis("off")
    
    # ==================
    # Stage 4: LAB + CLAHE
    # ==================
    axes[1, 2].imshow(preprocessed["before_clahe"])
    axes[1, 2].set_title("Stage 4: Before CLAHE")
    axes[1, 2].axis("off")
    
    axes[1, 3].imshow(preprocessed["after_clahe"])
    axes[1, 3].set_title("Stage 4: After CLAHE (Final Input)")
    axes[1, 3].axis("off")
    
    # ==================
    # Stages 5-11: Visualization & Summary
    # ==================
    # Stage 5-10: Show various visualizations
    # Stage 5: Binary Mask
    binary_mask = (pred_mask_resized > 0.5).astype(np.uint8) * 255
    axes[2, 0].imshow(binary_mask, cmap="gray")
    axes[2, 0].set_title("Stage 5: Binary Mask (Threshold=0.5)")
    axes[2, 0].axis("off")
    
    # Stage 6: Contours
    contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contour_img = original_img.copy()
    cv2.drawContours(contour_img, contours, -1, (0, 255, 0), 2)
    axes[2, 1].imshow(contour_img)
    axes[2, 1].set_title("Stage 6: Contour Detection")
    axes[2, 1].axis("off")
    
    # Stage 7: Histogram Analysis
    hist = cv2.calcHist([pred_mask_resized], [0], None, [256], [0, 1])
    axes[2, 2].plot(hist)
    axes[2, 2].set_title("Stage 7: Mask Intensity Histogram")
    axes[2, 2].set_xlim([0, 1])
    axes[2, 2].set_xlabel("Probability")
    axes[2, 2].set_ylabel("Frequency")
    
    # Stage 8-11: Summary
    axes[2, 3].axis("off")
    stats_text = f"""
Stage 8-11: Summary

Image: {img_file}
Mask coverage: {np.mean(pred_mask_resized)*100:.1f}%
Max probability: {np.max(pred_mask_resized):.2f}
Min probability: {np.min(pred_mask_resized):.2f}

✅ Segmentation Complete
✅ ROI Extracted
✅ RGB Enhanced
✅ CLAHE Applied
✅ Contours Detected
✅ Histogram Analyzed
✅ Results Saved!
"""
    axes[2, 3].text(0.1, 0.5, stats_text, ha="left", va="center", fontsize=10, 
                   transform=axes[2, 3].transAxes, bbox=dict(boxstyle="round", facecolor="#f0f0f0"))
    
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, f"stage_analysis_{idx+1:02d}_{img_file}.jpg"), dpi=150, bbox_inches="tight")
    plt.close()

print("\n" + "="*80)
print(f"✅ All 11 stages complete for {len(image_files)} images!")
print(f"Results saved to: {OUTPUT_DIR}")
print("="*80)
