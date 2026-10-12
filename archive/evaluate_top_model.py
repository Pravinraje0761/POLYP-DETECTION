
import os
import numpy as np
from PIL import Image
import torch
import albumentations as A
from albumentations.pytorch import ToTensorV2
import matplotlib.pyplot as plt
from adenseunet_simple_original import ADenseUNet
from polypgate.preprocessing.pipeline import complete_preprocessing_pipeline


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    output_dir = "top_model_visualizations"
    os.makedirs(output_dir, exist_ok=True)
    
    # Load our TOP model (90.86% Dice)
    print("=" * 60)
    print("LOADING TOP MODEL - 90.86% Dice Score")
    print("=" * 60)
    model = ADenseUNet(in_channels=3, out_channels=1).to(device)
    checkpoint_path = "checkpoints/best_adenseunet_boosted.pth"
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    print(f"Model loaded: {checkpoint_path}")
    print(f"Best Dice: {checkpoint['best_dice']:.4f}")
    print(f"Trained for {checkpoint['epoch']} epochs\n")
    
    # Process 15 samples for visual inspection
    data_root = "archive (1)/PNG"
    original_dir = os.path.join(data_root, "Original")
    gt_dir = os.path.join(data_root, "Ground Truth")
    image_files = sorted([f for f in os.listdir(original_dir) if f.endswith(".png")])[:15]
    
    total_dice = 0.0
    total_iou = 0.0
    
    for idx, img_file in enumerate(image_files):
        img_path = os.path.join(original_dir, img_file)
        gt_path = os.path.join(gt_dir, img_file)
        
        img = np.array(Image.open(img_path).convert("RGB"))
        gt_mask = np.array(Image.open(gt_path).convert("L"), dtype=np.float32) / 255.0
        
        # Run the full pipeline
        pp = complete_preprocessing_pipeline(img, model, device)
        
        # Calculate metrics for this sample
        pred_mask = pp["prob_mask"]
        intersection = np.sum(pred_mask * gt_mask)
        union = np.sum(pred_mask) + np.sum(gt_mask) - intersection
        dice = (2 * intersection + 1e-6) / (np.sum(pred_mask) + np.sum(gt_mask) + 1e-6)
        iou = (intersection + 1e-6) / (union + 1e-6)
        
        total_dice += dice
        total_iou += iou
        
        # Visualize
        fig, axes = plt.subplots(3, 4, figsize=(18, 12))
        
        # Row 0
        axes[0,0].imshow(img)
        axes[0,0].set_title("Original Image")
        axes[0,0].axis("off")
        
        axes[0,1].imshow(gt_mask, cmap="gray")
        axes[0,1].set_title("Ground Truth")
        axes[0,1].axis("off")
        
        axes[0,2].imshow(img)
        axes[0,2].imshow(gt_mask, cmap="jet", alpha=0.4)
        axes[0,2].set_title("GT Overlay")
        axes[0,2].axis("off")
        
        axes[0,3].imshow(pred_mask, cmap="gray")
        axes[0,3].set_title(f"Prediction (Dice: {dice:.3f})")
        axes[0,3].axis("off")
        
        # Row 1
        axes[1,0].imshow(img)
        axes[1,0].imshow(pred_mask, cmap="jet", alpha=0.4)
        axes[1,0].set_title("Prediction Overlay")
        axes[1,0].axis("off")
        
        axes[1,1].imshow(pp["roi"])
        axes[1,1].set_title("Stage 2: Soft ROI")
        axes[1,1].axis("off")
        
        axes[1,2].imshow(pp["blended_rgb"])
        axes[1,2].set_title("Stage 3: RGB Enhanced")
        axes[1,2].axis("off")
        
        axes[1,3].imshow(pp["after_clahe"])
        axes[1,3].set_title("Stage 4: Final (CLAHE)")
        axes[1,3].axis("off")
        
        # Row 2
        axes[2,0].axis("off")
        axes[2,1].text(0.5, 0.5, f"Image: {img_file}\nDice: {dice:.4f}\nIoU: {iou:.4f}", 
                        ha="center", va="center", fontsize=12, transform=axes[2,1].transAxes)
        axes[2,1].axis("off")
        axes[2,2].axis("off")
        axes[2,3].axis("off")
        
        plt.suptitle(f"TOP MODEL - {img_file} - Dice = {dice:.4f}", fontsize=16, y=0.98)
        plt.tight_layout()
        save_path = os.path.join(output_dir, f"sample_{idx+1:02d}_{img_file}")
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close()
        
        print(f"Processed {idx+1}/{len(image_files)}: {img_file} - Dice: {dice:.4f}")
    
    avg_dice = total_dice / len(image_files)
    avg_iou = total_iou / len(image_files)
    
    print("\n" + "=" * 60)
    print(f"AVERAGE RESULTS ON {len(image_files)} SAMPLES")
    print("=" * 60)
    print(f"Average Dice: {avg_dice:.4f}")
    print(f"Average IoU: {avg_iou:.4f}")
    print("\nVisualizations saved to:", output_dir)


if __name__ == "__main__":
    main()
