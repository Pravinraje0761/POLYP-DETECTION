
import os
import pandas as pd

# Load labels
df = pd.read_csv("hyper-kvasir-labeled-images (1)/labeled-images/image-labels.csv")

# Find polyp entries
polyp_entries = df[df['Finding'].str.contains('polyp', case=False)]
print(f"Found {len(polyp_entries)} polyp images!")

# Take first one
sample_entry = polyp_entries.iloc[0]
video_file = sample_entry['Video file']
print(f"Sample video file: {video_file}")

# Find the image path
data_root = "hyper-kvasir-labeled-images (1)/labeled-images/lower-gi-tract"
image_path = None
for root, dirs, files in os.walk(data_root):
    if f"{video_file}.jpg" in files:
        image_path = os.path.join(root, f"{video_file}.jpg")
        break

print(f"Image path: {image_path}")

# Now let's test the stages
import numpy as np
import cv2
from PIL import Image
from roi_extraction import soft_roi_extraction
from rgb_enhancement import selective_rgb_enhancement
from clahe import lab_clahe_enhancement

# Load image
if image_path:
    img = np.array(Image.open(image_path).convert("RGB"))
    print(f"Image shape: {img.shape}")
    
    # Create a dummy probability mask (for testing, since we don't have segmentation model yet)
    h, w = img.shape[:2]
    y, x = np.ogrid[:h, :w]
    center = (h//2, w//2)
    radius = min(h, w) // 4
    mask = np.exp(-((x - center[1])**2 + (y - center[0])**2) / (2 * radius**2))
    mask = (mask - mask.min()) / (mask.max() - mask.min())
    print(f"Mask shape: {mask.shape}")
    
    # Stage 2: Soft ROI Extraction
    orig, prob_mask, roi = soft_roi_extraction(img, mask)
    
    # Stage 3: Selective RGB Enhancement
    orig_rgb, enhanced_rgb, blended_rgb = selective_rgb_enhancement(img, mask)
    
    # Stage 4: LAB + CLAHE
    before_clahe, after_clahe = lab_clahe_enhancement(blended_rgb)
    
    # Save outputs
    os.makedirs("stage_outputs", exist_ok=True)
    
    # Save Stage 2
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    axes[0].imshow(orig)
    axes[0].set_title("Original Image")
    axes[0].axis('off')
    axes[1].imshow(prob_mask, cmap='gray')
    axes[1].set_title("Probability Mask (Dummy)")
    axes[1].axis('off')
    axes[2].imshow(roi)
    axes[2].set_title("Soft ROI Image")
    axes[2].axis('off')
    plt.tight_layout()
    plt.savefig("stage_outputs/stage2_soft_roi.png", dpi=300, bbox_inches='tight')
    print("Stage 2 output saved!")
    
    # Save Stage 3
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    axes[0].imshow(orig_rgb)
    axes[0].set_title("Original Image")
    axes[0].axis('off')
    axes[1].imshow(enhanced_rgb)
    axes[1].set_title("Enhanced ROI")
    axes[1].axis('off')
    axes[2].imshow(blended_rgb)
    axes[2].set_title("Blended Output")
    axes[2].axis('off')
    plt.tight_layout()
    plt.savefig("stage_outputs/stage3_rgb_enhancement.png", dpi=300, bbox_inches='tight')
    print("Stage 3 output saved!")
    
    # Save Stage 4
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    axes[0].imshow(before_clahe)
    axes[0].set_title("Before CLAHE")
    axes[0].axis('off')
    axes[1].imshow(after_clahe)
    axes[1].set_title("After CLAHE")
    axes[1].axis('off')
    plt.tight_layout()
    plt.savefig("stage_outputs/stage4_clahe.png", dpi=300, bbox_inches='tight')
    print("Stage 4 output saved!")
    
    print("All test outputs saved!")
