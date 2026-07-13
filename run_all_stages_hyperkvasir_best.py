
import os
import torch
import cv2
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from tqdm import tqdm
import albumentations as A
from albumentations.pytorch import ToTensorV2

# Import our modules
from adenseunet_boosted import ADenseUNet_Boosted
from preprocessing import (
    soft_roi_extraction,
    selective_rgb_enhancement,
    lab_clahe_enhancement
)

# Configuration
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
IMAGE_SIZE = 224
CHECKPOINT_PATH = "checkpoints/best_adenseunet_boosted.pth"
OUTPUT_DIR = "hyperkvasir_11_stages_best_model"
HYPERKVASIR_DIR = r"C:\Users\ADMIN\Desktop\POLYP_DETECTION\hyper-kvasir-labeled-images (1)\labeled-images\lower-gi-tract\pathological-findings\polyps"
NUM_IMAGES = 5  # Process first 5 images

# Create output directory
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Load the model
def load_model():
    print(f"Loading model from {CHECKPOINT_PATH}")
    model = ADenseUNet_Boosted(in_channels=3, out_channels=1).to(DEVICE)
    
    # Load checkpoint if available
    if os.path.exists(CHECKPOINT_PATH):
        try:
            checkpoint = torch.load(CHECKPOINT_PATH, map_location=DEVICE, weights_only=False)
            model.load_state_dict(checkpoint["model_state_dict"])
            print(f"Loaded model with best dice: {checkpoint.get('best_dice', 'N/A'):.6f}")
        except Exception as e:
            print(f"Error loading checkpoint: {e}, using untrained model")
    else:
        print(f"Checkpoint {CHECKPOINT_PATH} not found, using untrained model")
    
    model.eval()
    return model

# Transform for model input
transform = A.Compose([
    A.Resize(IMAGE_SIZE, IMAGE_SIZE),
    A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ToTensorV2(),
])

def process_single_image(img_path, model, idx):
    # 1. Load original image
    original_img = np.array(Image.open(img_path).convert("RGB"))
    
    # Get image for model input
    img_input = transform(image=original_img)["image"].unsqueeze(0).to(DEVICE)
    
    # 2. Perform segmentation
    with torch.no_grad():
        pred_logits = model(img_input)
        pred_mask = torch.sigmoid(pred_logits)[0, 0].cpu().numpy()
    
    # Resize mask to original image size
    pred_mask_resized = cv2.resize(pred_mask, (original_img.shape[1], original_img.shape[0]))
    
    # 3. ROI extraction
    _, _, roi_img = soft_roi_extraction(original_img, pred_mask_resized)
    
    # 4. Selective RGB enhancement
    _, enhanced_rgb, blended_rgb = selective_rgb_enhancement(original_img, pred_mask_resized)
    
    # 5. LAB + CLAHE
    before_clahe, after_clahe = lab_clahe_enhancement(blended_rgb)
    
    # 6. Binary mask
    binary_mask = (pred_mask_resized > 0.5).astype(np.uint8) * 255
    
    # 7. Contour detection
    contour_img = original_img.copy()
    contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(contour_img, contours, -1, (0, 255, 0), 2)
    
    # Create figure with 12 subplots (11 stages + summary)
    fig, axes = plt.subplots(3, 4, figsize=(20, 15))
    fig.suptitle(f"All 11 Stages (Best Model) - {os.path.basename(img_path)}", fontsize=16, y=0.98)
    axes = axes.flatten()
    
    # Stage 1: Original Image
    axes[0].imshow(original_img)
    axes[0].set_title("Stage 1: Original Image")
    axes[0].axis("off")
    
    # Stage 2: Segmentation Mask
    axes[1].imshow(pred_mask_resized, cmap="gray")
    axes[1].set_title("Stage 2: Segmentation Mask")
    axes[1].axis("off")
    
    # Stage 3: Segmentation Overlay
    axes[2].imshow(original_img)
    axes[2].imshow(pred_mask_resized, cmap="jet", alpha=0.4)
    axes[2].set_title("Stage 3: Segmentation Overlay")
    axes[2].axis("off")
    
    # Stage 4: ROI Extraction
    axes[3].imshow(roi_img)
    axes[3].set_title("Stage 4: ROI Extraction")
    axes[3].axis("off")
    
    # Stage 5: RGB Enhanced
    axes[4].imshow(enhanced_rgb)
    axes[4].set_title("Stage 5: RGB Enhanced")
    axes[4].axis("off")
    
    # Stage 6: RGB Blended
    axes[5].imshow(blended_rgb)
    axes[5].set_title("Stage 6: RGB Blended")
    axes[5].axis("off")
    
    # Stage 7: Before CLAHE
    axes[6].imshow(before_clahe)
    axes[6].set_title("Stage 7: Before CLAHE")
    axes[6].axis("off")
    
    # Stage 8: After CLAHE
    axes[7].imshow(after_clahe)
    axes[7].set_title("Stage 8: After CLAHE")
    axes[7].axis("off")
    
    # Stage 9: Binary Mask
    axes[8].imshow(binary_mask, cmap="gray")
    axes[8].set_title("Stage 9: Binary Mask")
    axes[8].axis("off")
    
    # Stage 10: Contour Detection
    axes[9].imshow(contour_img)
    axes[9].set_title("Stage 10: Contour Detection")
    axes[9].axis("off")
    
    # Stage 11: Mask Intensity Histogram
    axes[10].hist(pred_mask_resized.flatten(), bins=50, range=[0, 1], color="blue", alpha=0.7)
    axes[10].set_title("Stage 11: Mask Intensity Histogram")
    axes[10].set_xlabel("Probability")
    axes[10].set_ylabel("Frequency")
    
    # Summary panel
    axes[11].axis("off")
    summary_text = f"""
    Summary
    -------
    Image: {os.path.basename(img_path)}
    Mask Coverage: {np.mean(pred_mask_resized)*100:.1f}%
    Max Probability: {np.max(pred_mask_resized):.2f}
    Min Probability: {np.min(pred_mask_resized):.2f}
    Stages 1-11 Complete!
    """
    axes[11].text(0.1, 0.5, summary_text, ha="left", va="center", fontsize=10,
                  transform=axes[11].transAxes, bbox=dict(boxstyle="round", facecolor="#f0f0f0"))
    
    plt.tight_layout()
    output_path = os.path.join(OUTPUT_DIR, f"stage_analysis_{idx+1:02d}_{os.path.basename(img_path)}.png")
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    return output_path

def main():
    print("="*80)
    print("Running All 11 Stages on Hyper-Kvasir Images with BEST Model")
    print("="*80)
    
    # Load model
    model = load_model()
    
    # Get list of images
    image_files = sorted([f for f in os.listdir(HYPERKVASIR_DIR) 
                          if f.lower().endswith(('.jpg', '.jpeg', '.png'))])[:NUM_IMAGES]
    
    if not image_files:
        print("No images found in the directory!")
        return
    
    print(f"Found {len(image_files)} images. Processing...")
    
    # Process each image
    for idx, img_file in enumerate(tqdm(image_files, desc="Processing Images")):
        img_path = os.path.join(HYPERKVASIR_DIR, img_file)
        try:
            output_path = process_single_image(img_path, model, idx)
        except Exception as e:
            print(f"Error processing {img_file}: {e}")
    
    print("\n" + "="*80)
    print(f"Successfully processed {len(image_files)} images!")
    print(f"Results saved in: {os.path.abspath(OUTPUT_DIR)}")
    print("="*80)

if __name__ == "__main__":
    main()
