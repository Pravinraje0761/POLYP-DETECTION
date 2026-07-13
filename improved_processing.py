
import os
import numpy as np
from PIL import Image
import cv2
import torch
import albumentations as A
from albumentations.pytorch import ToTensorV2
import matplotlib.pyplot as plt
from adenseunet_simple import ADenseUNet
from vit_classifier import PolypViT
from preprocessing import soft_roi_extraction, selective_rgb_enhancement, lab_clahe_enhancement

def preprocess_for_segmentation(image, image_size=224):
    """
    Improved preprocessing for segmentation with better histogram matching
    and normalization.
    """
    # Resize
    img_resized = cv2.resize(image, (image_size, image_size))
    
    # Convert to RGB if needed
    if len(img_resized.shape) == 2:
        img_resized = cv2.cvtColor(img_resized, cv2.COLOR_GRAY2RGB)
    elif img_resized.shape[-1] == 4:
        img_resized = cv2.cvtColor(img_resized, cv2.COLOR_RGBA2RGB)
    
    # Apply CLAHE on each channel for better contrast
    img_ycrcb = cv2.cvtColor(img_resized, cv2.COLOR_RGB2YCrCb)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    img_ycrcb[:, :, 0] = clahe.apply(img_ycrcb[:, :, 0])
    img_enhanced = cv2.cvtColor(img_ycrcb, cv2.COLOR_YCrCb2RGB)
    
    return img_enhanced, img_resized

def get_segmentation_mask(image, model, device, image_size=224):
    """
    Get segmentation mask with improved post-processing
    """
    # Get original size
    orig_h, orig_w = image.shape[:2]
    
    # Preprocess
    img_preprocessed, img_resized = preprocess_for_segmentation(image, image_size)
    
    # Transform to tensor
    transform = A.Compose([
        A.Resize(height=image_size, width=image_size),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2(),
    ])
    input_tensor = transform(image=img_preprocessed)["image"].unsqueeze(0).to(device)
    
    # Get prediction
    model.eval()
    with torch.no_grad():
        output = model(input_tensor)
        prob_mask = torch.sigmoid(output).squeeze().cpu().numpy()
    
    # Resize back to original size
    prob_mask = cv2.resize(prob_mask, (orig_w, orig_h))
    
    # Apply post-processing (gaussian blur to smooth)
    prob_mask = cv2.GaussianBlur(prob_mask, (5, 5), 0)
    
    return prob_mask

def main():
    # Configuration
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    polyp_folder = r"C:\Users\ADMIN\Desktop\POLYP_DETECTION\hyper-kvasir-labeled-images (1)\labeled-images\lower-gi-tract\pathological-findings\polyps"
    training_folder = r"C:\Users\ADMIN\Desktop\POLYP_DETECTION\archive (1)\PNG"
    output_dir = "improved_results"
    os.makedirs(output_dir, exist_ok=True)
    
    # Load segmentation model
    seg_model = ADenseUNet(in_channels=3, out_channels=1).to(device)
    seg_checkpoint = torch.load("checkpoints/best_adenseunet.pth", map_location=device, weights_only=False)
    seg_model.load_state_dict(seg_checkpoint["model_state_dict"])
    seg_model.eval()
    
    # Load classification model
    cls_model = PolypViT(num_classes=2, pretrained=False).to(device)
    cls_checkpoint = torch.load("outputs/checkpoints/best_model.pth", map_location=device, weights_only=False)
    cls_model.load_state_dict(cls_checkpoint["model_state_dict"])
    cls_model.eval()
    
    # Process first 5 training images first to verify segmentation
    print("Processing training images to verify segmentation...")
    train_images = sorted([f for f in os.listdir(os.path.join(training_folder, "Original")) if f.endswith(".png")])[:5]
    
    for i, img_file in enumerate(train_images):
        img_path = os.path.join(training_folder, "Original", img_file)
        gt_path = os.path.join(training_folder, "Ground Truth", img_file)
        
        img = np.array(Image.open(img_path).convert("RGB"))
        gt_mask = np.array(Image.open(gt_path).convert("L")) / 255.0
        
        # Get our prediction
        pred_mask = get_segmentation_mask(img, seg_model, device)
        
        # Visualize
        fig, axes = plt.subplots(1, 4, figsize=(16, 4))
        axes[0].imshow(img)
        axes[0].set_title("Original")
        axes[0].axis('off')
        
        axes[1].imshow(gt_mask, cmap='gray')
        axes[1].set_title("GT Mask")
        axes[1].axis('off')
        
        axes[2].imshow(pred_mask, cmap='gray')
        axes[2].set_title("Predicted Mask")
        axes[2].axis('off')
        
        axes[3].imshow(img)
        axes[3].imshow(pred_mask, cmap='jet', alpha=0.4)
        axes[3].set_title("Overlay")
        axes[3].axis('off')
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"train_verify_{i+1}.png"), dpi=150, bbox_inches="tight")
        plt.close()
    
    print("Training verification done! Now processing Hyper-Kvasir images...")
    
    # Get 10 Hyper-Kvasir polyp images
    image_files = sorted([f for f in os.listdir(polyp_folder) if f.endswith(".jpg")])[:10]
    
    for i, img_file in enumerate(image_files):
        print(f"Processing image {i+1}/10: {img_file}")
        
        img_path = os.path.join(polyp_folder, img_file)
        img = np.array(Image.open(img_path).convert("RGB"))
        
        # Segmentation
        prob_mask = get_segmentation_mask(img, seg_model, device)
        
        # Stages 2, 3, 4
        orig, soft_mask, roi_img = soft_roi_extraction(img, prob_mask)
        orig_rgb, enhanced_rgb, blended_rgb = selective_rgb_enhancement(img, prob_mask)
        before_clahe, after_clahe = lab_clahe_enhancement(blended_rgb)
        
        # Visualize with larger figure
        fig, axes = plt.subplots(2, 4, figsize=(20, 10))
        
        # Row 1
        axes[0, 0].imshow(img)
        axes[0, 0].set_title("1. Original", fontsize=12)
        axes[0, 0].axis('off')
        
        axes[0, 1].imshow(prob_mask, cmap='gray')
        axes[0, 1].set_title("2. Segmentation Mask", fontsize=12)
        axes[0, 1].axis('off')
        
        axes[0, 2].imshow(roi_img)
        axes[0, 2].set_title("3. Soft ROI", fontsize=12)
        axes[0, 2].axis('off')
        
        axes[0, 3].imshow(blended_rgb)
        axes[0, 3].set_title("4. RGB Enhanced", fontsize=12)
        axes[0, 3].axis('off')
        
        # Row 2
        axes[1, 0].imshow(after_clahe)
        axes[1, 0].set_title("5. Final Preprocessed", fontsize=12)
        axes[1, 0].axis('off')
        
        axes[1, 1].imshow(img)
        axes[1, 1].imshow(prob_mask > 0.5, cmap='jet', alpha=0.4)
        axes[1, 1].set_title("6. Mask Overlay (T=0.5)", fontsize=12)
        axes[1, 1].axis('off')
        
        axes[1, 2].imshow(img)
        axes[1, 2].imshow(prob_mask > 0.3, cmap='jet', alpha=0.4)
        axes[1, 2].set_title("7. Mask Overlay (T=0.3)", fontsize=12)
        axes[1, 2].axis('off')
        
        axes[1, 3].imshow(img)
        axes[1, 3].imshow(prob_mask > 0.7, cmap='jet', alpha=0.4)
        axes[1, 3].set_title("8. Mask Overlay (T=0.7)", fontsize=12)
        axes[1, 3].axis('off')
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"polyp_improved_{i+1:02d}.png"), dpi=150, bbox_inches="tight")
        plt.close()
    
    print("All done! Results saved to improved_results/")

if __name__ == "__main__":
    main()
