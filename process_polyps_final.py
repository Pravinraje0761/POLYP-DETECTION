
"""
Process polyp images using our best ADenseUNet model (89.87% Dice)
"""
import os
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend to avoid issues
import matplotlib.pyplot as plt
from PIL import Image
from tqdm import tqdm
import argparse
import warnings
warnings.filterwarnings("ignore")

# Import modules (use original ADenseUNet)
from adenseunet_simple_original import ADenseUNet
from vit_classifier import PolypViT
from roi_extraction import soft_roi_extraction
from rgb_enhancement import selective_rgb_enhancement
from clahe import lab_clahe_enhancement
from gradcam import GradCAM, visualize_gradcam
from attention_rollout import attention_rollout, visualize_attention_rollout
from albumentations import Compose, Resize, Normalize
from albumentations.pytorch import ToTensorV2

def parse_args():
    parser = argparse.ArgumentParser(description="Process polyp images with best model")
    parser.add_argument("--input_folder", type=str, default=r"C:\Users\ADMIN\Desktop\POLYP_DETECTION\hyper-kvasir-labeled-images (1)\labeled-images\lower-gi-tract\pathological-findings\polyps", help="Folder with polyp images")
    parser.add_argument("--output_folder", type=str, default=r"C:\Users\ADMIN\Desktop\POLYP_DETECTION\final_output", help="Folder to save outputs")
    parser.add_argument("--seg_checkpoint", type=str, default="checkpoints/best_adenseunet_ultimate.pth", help="Segmentation checkpoint path")
    parser.add_argument("--cls_checkpoint", type=str, default="outputs/checkpoints/best_model.pth", help="Classification checkpoint path")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--num_samples", type=int, default=20, help="Number of samples to process")
    return parser.parse_args()

def main():
    args = parse_args()
    device = torch.device(args.device)
    os.makedirs(args.output_folder, exist_ok=True)
    for subfolder in ["segmentation_masks", "roi_extraction", "rgb_enhancement", "lab_clahe", "gradcam", "attention_rollout", "all_stages_combined"]:
        os.makedirs(os.path.join(args.output_folder, subfolder), exist_ok=True)

    print("="*80)
    print("PROCESSING POLYP IMAGES WITH BEST MODEL (89.87% DICE)")
    print("="*80)

    # Load segmentation model
    print("\n[1] Loading segmentation model...")
    seg_model = ADenseUNet(in_channels=3, out_channels=1).to(device)
    try:
        seg_checkpoint = torch.load(args.seg_checkpoint, map_location=device, weights_only=False)
        seg_model.load_state_dict(seg_checkpoint["model_state_dict"])
        print(f"    [OK] Loaded checkpoint from {args.seg_checkpoint}")
        print(f"    [OK] Best Dice: {seg_checkpoint['best_dice']:.4f}")
    except Exception as e:
        print(f"    [ERROR] Loading checkpoint: {e}")
        print("    [WARN] Using untrained model - results will be poor!")
    seg_model.eval()

    # Load classification model
    print("\n[2] Loading classification model...")
    cls_model = None
    try:
        cls_model = PolypViT(num_classes=2, pretrained=True).to(device)
        if os.path.exists(args.cls_checkpoint):
            cls_checkpoint = torch.load(args.cls_checkpoint, map_location=device, weights_only=False)
            cls_model.load_state_dict(cls_checkpoint["model_state_dict"])
            print(f"    [OK] Loaded classification checkpoint")
        else:
            print(f"    [WARN] Classification checkpoint not found - using pretrained ViT")
        cls_model.eval()
    except Exception as e:
        print(f"    [ERROR] Loading classification model: {e}")

    # Get image list
    image_paths = []
    for root, dirs, files in os.walk(args.input_folder):
        for file in files:
            if file.lower().endswith((".jpg", ".jpeg", ".png")):
                image_paths.append(os.path.join(root, file))
    print(f"\n[3] Found {len(image_paths)} polyp images")
    num_process = min(args.num_samples, len(image_paths))
    print(f"    Processing {num_process} samples...")

    # Process images
    for idx, img_path in enumerate(tqdm(image_paths[:num_process], desc="Processing")):
        img_name = os.path.basename(img_path)
        img = np.array(Image.open(img_path).convert("RGB"))
        orig_h, orig_w = img.shape[:2]

        # --------------------------
        # [1] Segmentation
        # --------------------------
        seg_transform = Compose([
            Resize(352, 352),
            Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2(),
        ])
        seg_input = seg_transform(image=img)["image"].unsqueeze(0).to(device)
        
        with torch.no_grad():
            seg_output = seg_model(seg_input)
            seg_mask = torch.sigmoid(seg_output).squeeze().cpu().numpy()
        
        # Resize mask to original size
        seg_mask_pil = Image.fromarray((seg_mask * 255).astype(np.uint8)).resize((orig_w, orig_h), Image.Resampling.LANCZOS)
        seg_mask_orig = np.array(seg_mask_pil) / 255.0
        
        # Save mask
        mask_path = os.path.join(args.output_folder, "segmentation_masks", f"mask_{img_name}")
        Image.fromarray((seg_mask_orig * 255).astype(np.uint8)).save(mask_path)

        # --------------------------
        # [2] ROI Extraction
        # --------------------------
        _, _, roi_img = soft_roi_extraction(img, seg_mask_orig)
        roi_save = os.path.join(args.output_folder, "roi_extraction", f"roi_{img_name}")
        fig, axes = plt.subplots(1, 3, figsize=(12, 6))
        axes[0].imshow(img), axes[0].set_title("Original"), axes[0].axis("off")
        axes[1].imshow(seg_mask_orig, cmap="gray"), axes[1].set_title("Segmentation"), axes[1].axis("off")
        axes[2].imshow(roi_img), axes[2].set_title("ROI"), axes[2].axis("off")
        plt.tight_layout(), plt.savefig(roi_save, dpi=100, bbox_inches="tight"), plt.close()

        # --------------------------
        # [3] RGB Enhancement
        # --------------------------
        _, enhanced_rgb, blended_rgb = selective_rgb_enhancement(roi_img, seg_mask_orig)
        rgb_save = os.path.join(args.output_folder, "rgb_enhancement", f"rgb_{img_name}")
        fig, axes = plt.subplots(1, 3, figsize=(12, 6))
        axes[0].imshow(roi_img), axes[0].set_title("ROI"), axes[0].axis("off")
        axes[1].imshow(enhanced_rgb), axes[1].set_title("RGB Enhanced"), axes[1].axis("off")
        axes[2].imshow(blended_rgb), axes[2].set_title("Blended"), axes[2].axis("off")
        plt.tight_layout(), plt.savefig(rgb_save, dpi=100, bbox_inches="tight"), plt.close()

        # --------------------------
        # [4] LAB-CLAHE
        # --------------------------
        _, after_clahe = lab_clahe_enhancement(blended_rgb)
        clahe_save = os.path.join(args.output_folder, "lab_clahe", f"clahe_{img_name}")
        fig, axes = plt.subplots(1, 2, figsize=(12, 6))
        axes[0].imshow(blended_rgb), axes[0].set_title("Before CLAHE"), axes[0].axis("off")
        axes[1].imshow(after_clahe), axes[1].set_title("After CLAHE"), axes[1].axis("off")
        plt.tight_layout(), plt.savefig(clahe_save, dpi=100, bbox_inches="tight"), plt.close()

        # --------------------------
        # [5] Grad-CAM + Attention Rollout
        # --------------------------
        gradcam_overlay = None
        attn_overlay = None
        
        if cls_model is not None:
            try:
                # Grad-CAM
                cls_transform = Compose([
                    Resize(224, 224),
                    Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                    ToTensorV2(),
                ])
                cls_input = cls_transform(image=after_clahe)["image"].unsqueeze(0).to(device)
                
                gradcam = GradCAM(cls_model, cls_model.vit.encoder.layers[-1].ln_1)
                cam = gradcam.generate_cam(cls_input, target_class=1)
                gradcam_overlay, gradcam_heatmap = visualize_gradcam(after_clahe, cam, 224)
                
                gradcam_save = os.path.join(args.output_folder, "gradcam", f"gradcam_{img_name}")
                fig, axes = plt.subplots(1, 3, figsize=(15, 5))
                axes[0].imshow(after_clahe), axes[0].set_title("Enhanced"), axes[0].axis("off")
                axes[1].imshow(gradcam_heatmap, cmap="jet"), axes[1].set_title("Grad-CAM Heatmap"), axes[1].axis("off")
                axes[2].imshow(gradcam_overlay), axes[2].set_title("Grad-CAM"), axes[2].axis("off")
                plt.tight_layout(), plt.savefig(gradcam_save, dpi=100, bbox_inches="tight"), plt.close()

                # Attention Rollout
                with torch.no_grad():
                    _, attn_weights = cls_model(cls_input)
                rollout = attention_rollout(attn_weights)
                attn_overlay, attn_heatmap = visualize_attention_rollout(after_clahe, rollout, 224)
                
                attn_save = os.path.join(args.output_folder, "attention_rollout", f"attn_{img_name}")
                fig, axes = plt.subplots(1, 3, figsize=(15, 5))
                axes[0].imshow(after_clahe), axes[0].set_title("Enhanced"), axes[0].axis("off")
                axes[1].imshow(attn_heatmap, cmap="jet"), axes[1].set_title("Attention Heatmap"), axes[1].axis("off")
                axes[2].imshow(attn_overlay), axes[2].set_title("Attention"), axes[2].axis("off")
                plt.tight_layout(), plt.savefig(attn_save, dpi=100, bbox_inches="tight"), plt.close()
            except Exception as e:
                print(f"\n    [WARN] Classification error for {img_name}: {e}")

        # --------------------------
        # [6] Combined Visualization
        # --------------------------
        combined_save = os.path.join(args.output_folder, "all_stages_combined", f"combined_{img_name}")
        fig, axes = plt.subplots(2, 4, figsize=(20, 10))
        axes = axes.flatten()
        
        axes[0].imshow(img), axes[0].set_title("1. Original"), axes[0].axis("off")
        axes[1].imshow(seg_mask_orig, cmap="gray"), axes[1].set_title("2. Segmentation"), axes[1].axis("off")
        axes[2].imshow(roi_img), axes[2].set_title("3. ROI"), axes[2].axis("off")
        axes[3].imshow(enhanced_rgb), axes[3].set_title("4. RGB Enhanced"), axes[3].axis("off")
        axes[4].imshow(blended_rgb), axes[4].set_title("5. Blended"), axes[4].axis("off")
        axes[5].imshow(after_clahe), axes[5].set_title("6. LAB-CLAHE"), axes[5].axis("off")
        
        if gradcam_overlay is not None:
            axes[6].imshow(gradcam_overlay), axes[6].set_title("7. Grad-CAM"), axes[6].axis("off")
        if attn_overlay is not None:
            axes[7].imshow(attn_overlay), axes[7].set_title("8. Attention Rollout"), axes[7].axis("off")
        
        plt.suptitle(f"Processing: {img_name}", fontsize=16), plt.tight_layout()
        plt.savefig(combined_save, dpi=150, bbox_inches="tight"), plt.close()

    print("\n" + "="*80)
    print("PROCESSING COMPLETE!")
    print("="*80)
    print(f"Outputs saved to: {args.output_folder}")
    print("\nFolders created:")
    print("  - segmentation_masks/ : Binary masks")
    print("  - roi_extraction/    : ROI visualization")
    print("  - rgb_enhancement/   : RGB enhancement")
    print("  - lab_clahe/         : CLAHE enhancement")
    print("  - gradcam/           : Grad-CAM explainability")
    print("  - attention_rollout/ : Attention maps")
    print("  - all_stages_combined/ : Full pipeline")


if __name__ == "__main__":
    main()
