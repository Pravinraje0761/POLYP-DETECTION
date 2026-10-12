
"""
Generate sample outputs for all stages using the archive (1) dataset
"""
import os
import numpy as np
from PIL import Image
import torch
import matplotlib.pyplot as plt
from adenseunet_simple import ADenseUNet
from polypgate.preprocessing.pipeline import complete_preprocessing_pipeline


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    output_dir = "archive_dataset_sample_outputs"
    os.makedirs(output_dir, exist_ok=True)
    
    # Load segmentation model
    seg_model = ADenseUNet(in_channels=3, out_channels=1).to(device)
    
    # Try to load the best model (improved first, then original)
    try:
        checkpoint = torch.load("checkpoints/best_adenseunet_improved.pth", 
                               map_location=device, weights_only=False)
        seg_model.load_state_dict(checkpoint["model_state_dict"])
        print("Using improved segmentation model")
    except:
        checkpoint = torch.load("checkpoints/best_adenseunet.pth", 
                               map_location=device, weights_only=False)
        seg_model.load_state_dict(checkpoint["model_state_dict"])
        print("Using original segmentation model")
        
    seg_model.eval()
    
    # Get sample images from archive (1) dataset
    data_root = "archive (1)/PNG"
    original_dir = os.path.join(data_root, "Original")
    gt_dir = os.path.join(data_root, "Ground Truth")
    image_files = sorted([f for f in os.listdir(original_dir) if f.endswith(".png")])[:10]
    
    for idx, img_file in enumerate(image_files):
        print(f"Processing sample {idx+1}/{len(image_files)}: {img_file}")
        img_path = os.path.join(original_dir, img_file)
        gt_path = os.path.join(gt_dir, img_file)
        
        img = np.array(Image.open(img_path).convert("RGB"))
        gt_mask = np.array(Image.open(gt_path).convert("L")) / 255.0
        
        # Run complete preprocessing
        pp = complete_preprocessing_pipeline(img, seg_model, device)
        
        # Create detailed visualization
        fig, axes = plt.subplots(3, 4, figsize=(18, 14))
        
        # Row 0: Original and Ground Truth
        axes[0, 0].imshow(img)
        axes[0, 0].set_title("Original Image")
        axes[0, 0].axis('off')
        
        axes[0, 1].imshow(gt_mask, cmap='gray')
        axes[0, 1].set_title("Ground Truth Mask")
        axes[0, 1].axis('off')
        
        axes[0, 2].imshow(img)
        axes[0, 2].imshow(gt_mask, cmap='jet', alpha=0.4)
        axes[0, 2].set_title("GT Overlay")
        axes[0, 2].axis('off')
        
        axes[0, 3].imshow(pp["prob_mask"], cmap='gray')
        axes[0, 3].set_title("Predicted Segmentation Mask")
        axes[0, 3].axis('off')
        
        # Row 1: Stage 2 & 3
        axes[1, 0].imshow(img)
        axes[1, 0].imshow(pp["prob_mask"], cmap='jet', alpha=0.4)
        axes[1, 0].set_title("Prediction Overlay")
        axes[1, 0].axis('off')
        
        axes[1, 1].imshow(pp["roi"])
        axes[1, 1].set_title("Stage 2: Soft ROI")
        axes[1, 1].axis('off')
        
        axes[1, 2].imshow(pp["enhanced_rgb"])
        axes[1, 2].set_title("Stage 3: Enhanced ROI")
        axes[1, 2].axis('off')
        
        axes[1, 3].imshow(pp["blended_rgb"])
        axes[1, 3].set_title("Stage 3: Blended Enhancement")
        axes[1, 3].axis('off')
        
        # Row 2: Stage 4
        axes[2, 0].imshow(pp["before_clahe"])
        axes[2, 0].set_title("Stage 4: Before CLAHE")
        axes[2, 0].axis('off')
        
        axes[2, 1].imshow(pp["after_clahe"])
        axes[2, 1].set_title("Stage 4: After CLAHE (Final Input)")
        axes[2, 1].axis('off')
        
        # Empty axes for spacing
        axes[2, 2].axis('off')
        axes[2, 3].axis('off')
        
        plt.tight_layout(rect=[0, 0.01, 1, 0.95])
        plt.suptitle(f"Complete Pipeline Demo - {img_file}", fontsize=16, y=0.98)
        save_path = os.path.join(output_dir, f"sample_{idx+1:02d}_{img_file}")
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"Saved to {save_path}")
    
    print(f"\nAll samples processed! Results in {output_dir}/")


if __name__ == "__main__":
    main()
