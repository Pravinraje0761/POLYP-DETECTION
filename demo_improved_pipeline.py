
"""
Demo script for the improved complete pipeline
Visualizes all stages for sample images
"""
import os
import numpy as np
from PIL import Image
import torch
import albumentations as A
from albumentations.pytorch import ToTensorV2
import matplotlib.pyplot as plt
import cv2
from adenseunet_simple import ADenseUNet
from end_to_end_model import EndToEndPolypModel
from preprocessing import complete_preprocessing_pipeline


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    output_dir = "improved_demo_outputs"
    os.makedirs(output_dir, exist_ok=True)
    
    # Load models
    print("Loading improved segmentation model...")
    seg_model = ADenseUNet(in_channels=3, out_channels=1).to(device)
    try:
        seg_checkpoint = torch.load("checkpoints/best_adenseunet_improved.pth", 
                                   map_location=device, weights_only=False)
        seg_model.load_state_dict(seg_checkpoint["model_state_dict"])
        print("Successfully loaded improved segmentation model!")
    except:
        print("Improved model not ready yet, using original best model...")
        seg_checkpoint = torch.load("checkpoints/best_adenseunet.pth", 
                                   map_location=device, weights_only=False)
        seg_model.load_state_dict(seg_checkpoint["model_state_dict"])
    seg_model.eval()
    
    # Process sample images from Hyper-Kvasir
    polyp_dir = r"C:\Users\ADMIN\Desktop\POLYP_DETECTION\hyper-kvasir-labeled-images (1)\labeled-images\lower-gi-tract\pathological-findings\polyps"
    sample_files = sorted([f for f in os.listdir(polyp_dir) if f.endswith('.jpg')])[:10]
    
    for idx, img_file in enumerate(sample_files):
        print(f"Processing sample {idx+1}/{len(sample_files)}: {img_file}")
        img_path = os.path.join(polyp_dir, img_file)
        img = np.array(Image.open(img_path).convert("RGB"))
        
        # Run complete preprocessing
        pp = complete_preprocessing_pipeline(img, seg_model, device)
        
        # Create visualization
        fig, axes = plt.subplots(2, 4, figsize=(16, 10))
        
        # Row 1
        axes[0, 0].imshow(img)
        axes[0, 0].set_title("Original Image")
        axes[0, 0].axis('off')
        
        axes[0, 1].imshow(pp["prob_mask"], cmap='gray')
        axes[0, 1].set_title("Segmentation Mask")
        axes[0, 1].axis('off')
        
        axes[0, 2].imshow(pp["roi"])
        axes[0, 2].set_title("Soft ROI")
        axes[0, 2].axis('off')
        
        axes[0, 3].imshow(pp["blended_rgb"])
        axes[0, 3].set_title("RGB Enhanced")
        axes[0, 3].axis('off')
        
        # Row 2
        axes[1, 0].imshow(pp["before_clahe"])
        axes[1, 0].set_title("Before CLAHE")
        axes[1, 0].axis('off')
        
        axes[1, 1].imshow(pp["after_clahe"])
        axes[1, 1].set_title("After CLAHE (Final Input)")
        axes[1, 1].axis('off')
        
        # Overlay of mask
        axes[1, 2].imshow(img)
        axes[1, 2].imshow(pp["prob_mask"] > 0.5, cmap='jet', alpha=0.4)
        axes[1, 2].set_title("Mask Overlay (Thresh=0.5)")
        axes[1, 2].axis('off')
        
        # Overlay with different threshold
        axes[1, 3].imshow(img)
        axes[1, 3].imshow(pp["prob_mask"] > 0.3, cmap='jet', alpha=0.4)
        axes[1, 3].set_title("Mask Overlay (Thresh=0.3)")
        axes[1, 3].axis('off')
        
        plt.tight_layout()
        save_path = os.path.join(output_dir, f"sample_{idx+1:02d}.png")
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"Saved to {save_path}")
    
    print(f"\nAll samples processed! Results in {output_dir}/")


if __name__ == "__main__":
    main()
