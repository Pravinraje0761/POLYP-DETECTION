
"""
Process polyp images using our best ADenseUNet model (89.87% Dice)
Generates:
- Segmentation masks
- ROI extraction
- RGB enhancement
- LAB-CLAHE
- Visualizations
- Grad-CAM
- Attention Rollout (if classification model available)
"""
import os
import numpy as np
import torch
import matplotlib.pyplot as plt
from PIL import Image
from tqdm import tqdm
import argparse

# Import our modules
from adenseunet_simple import ADenseUNet
from polypgate.models.vit_classifier import PolypViT
from polypgate.preprocessing.roi import soft_roi_extraction
from polypgate.preprocessing.rgb import selective_rgb_enhancement
from polypgate.preprocessing.clahe import lab_clahe_enhancement
from polypgate.explain.gradcam import GradCAM, visualize_gradcam
from polypgate.explain.attention_rollout import attention_rollout, visualize_attention_rollout
from albumentations import Compose, Resize, Normalize
from albumentations.pytorch import ToTensorV2

def parse_args():
    parser = argparse.ArgumentParser(description="Process polyp images with best model")
    parser.add_argument("--input_folder", type=str, default=r"C:\Users\ADMIN\Desktop\POLYP_DETECTION\hyper-kvasir-labeled-images (1)\labeled-images\lower-gi-tract\pathological-findings\polyps", help="Folder with polyp images")
    parser.add_argument("--output_folder", type=str, default=r"C:\Users\ADMIN\Desktop\POLYP_DETECTION\final_output", help="Folder to save outputs")
    parser.add_argument("--seg_checkpoint", type=str, default="checkpoints/best_adenseunet_ultimate.pth", help="Segmentation checkpoint path")
    parser.add_argument("--cls_checkpoint", type=str, default="outputs/checkpoints/best_model.pth", help="Classification checkpoint path")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--num_samples", type=int, default=20, help="Number of samples to process (for visualization)")
    return parser.parse_args()

def main():
    args = parse_args()
    device = torch.device(args.device)
    os.makedirs(args.output_folder, exist_ok=True)
    for subfolder in ["segmentation_masks", "roi_extraction", "rgb_enhancement", "lab_clahe", "gradcam", "attention_rollout", "all_stages_combined"]:
        os.makedirs(os.path.join(args.output_folder, subfolder), exist_ok=True)

    print("=" * 80)
    print("PROCESSING POLYP IMAGES WITH BEST MODEL (89.87% DICE)")
    print("=" * 80)

    # Load segmentation model
    print("\n[1] Loading segmentation model...")
    seg_model = ADenseUNet(in_channels=3, out_channels=1).to(device)
    try:
        seg_checkpoint = torch.load(args.seg_checkpoint, map_location=device, weights_only=False)
        # Fix key names (final_up vs up_final)
        state_dict = seg_checkpoint["model_state_dict"]
        new_state_dict = {}
        for key, value in state_dict.items():
            if key.startswith("final_up"):
                new_key = key.replace("final_up", "up_final")
                new_state_dict[new_key] = value
            else:
                new_state_dict[key] = value
        seg_model.load_state_dict(new_state_dict, strict=False)
        print(f"    ✓ Loaded checkpoint from {args.seg_checkpoint}")
        print(f"    ✓ Best Dice: {seg_checkpoint['best_dice']:.4f}")
    except Exception as e:
        print(f"    ⚠️  Error loading checkpoint: {e}")
        print("    Using untrained model (will not work well!)")
    seg_model.eval()

    # Load classification model (if available)
    print("\n[2] Loading classification model...")
    cls_model = None
    try:
        cls_model = PolypViT(num_classes=2, pretrained=True).to(device)
        if os.path.exists(args.cls_checkpoint):
            cls_checkpoint = torch.load(args.cls_checkpoint, map_location=device, weights_only=False)
            cls_model.load_state_dict(cls_checkpoint["model_state_dict"])
            print(f"    ✓ Loaded checkpoint from {args.cls_checkpoint}")
        else:
            print(f"    ⚠️  Classification checkpoint not found at {args.cls_checkpoint}")
            print(f"    Using pretrained ViT without fine-tuning")
        cls_model.eval()
    except Exception as e:
        print(f"    ⚠️  Error loading classification model: {e}")

    # Get list of polyp images
    image_paths = []
    for root, dirs, files in os.walk(args.input_folder):
        for file in files:
            if file.lower().endswith((".jpg", ".jpeg", ".png")):
                image_paths.append(os.path.join(root, file))
    print(f"\n[3] Found {len(image_paths)} polyp images")
    
    num_process = min(args.num_samples, len(image_paths))
    print(f"\n[4] Processing {num_process} samples...")

    # Process each image
    for idx, img_path in enumerate(tqdm(image_paths[:num_process], desc="Processing")):
        # Load image
        img_name = os.path.basename(img_path)
        img = np.array(Image.open(img_path).convert("RGB"))
        original_h, original_w = img.shape[:2]

        # Preprocessing for segmentation
        seg_transform = Compose([
            Resize(352, 352),
            Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2(),
        ])
        seg_input = seg_transform(image=img)["image"].unsqueeze(0).to(device)

        # Get segmentation mask
        with torch.no_grad():
            seg_output = seg_model(seg_input)
            seg_mask = torch.sigmoid(seg_output).squeeze().cpu().numpy()
            # Resize back to original size
            seg_mask_pil = Image.fromarray((seg_mask * 255).astype(np.uint8)).resize((original_w, original_h), Image.Resampling.LANCZOS)
            seg_mask = np.array(seg_mask_pil) / 255.0

        # Save segmentation mask
        mask_save_path = os.path.join(args.output_folder, "segmentation_masks", f"mask_{img_name}")
        Image.fromarray((seg_mask * 255).astype(np.uint8)).save(mask_save_path)

        # ROI extraction
        roi_img, roi_mask = soft_roi_extraction(img, seg_mask)
        roi_save_path = os.path.join(args.output_folder, "roi_extraction", f"roi_{img_name}")
        plt.figure(figsize=(12, 6))
        plt.subplot(131), plt.imshow(img), plt.title("Original"), plt.axis("off")
        plt.subplot(132), plt.imshow(seg_mask, cmap="gray"), plt.title("Segmentation Mask"), plt.axis("off")
        plt.subplot(133), plt.imshow(roi_img), plt.title("ROI Extraction"), plt.axis("off")
        plt.savefig(roi_save_path, dpi=100, bbox_inches="tight")
        plt.close()

        # RGB enhancement
        enhanced_rgb, blended_rgb = selective_rgb_enhancement(roi_img)
        rgb_save_path = os.path.join(args.output_folder, "rgb_enhancement", f"rgb_{img_name}")
        plt.figure(figsize=(12, 6))
        plt.subplot(131), plt.imshow(roi_img), plt.title("ROI Image"), plt.axis("off")
        plt.subplot(132), plt.imshow(enhanced_rgb), plt.title("RGB Enhanced"), plt.axis("off")
        plt.subplot(133), plt.imshow(blended_rgb), plt.title("Blended RGB"), plt.axis("off")
        plt.savefig(rgb_save_path, dpi=100, bbox_inches="tight")
        plt.close()

        # LAB-CLAHE
        after_clahe = lab_clahe_enhancement(blended_rgb)
        clahe_save_path = os.path.join(args.output_folder, "lab_clahe", f"clahe_{img_name}")
        plt.figure(figsize=(12, 6))
        plt.subplot(121), plt.imshow(blended_rgb), plt.title("Before CLAHE"), plt.axis("off")
        plt.subplot(122), plt.imshow(after_clahe), plt.title("After LAB-CLAHE"), plt.axis("off")
        plt.savefig(clahe_save_path, dpi=100, bbox_inches="tight")
        plt.close()

        # Grad-CAM (if classification model exists)
        if cls_model is not None:
            try:
                cls_transform = Compose([
                    Resize(224, 224),
                    Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                    ToTensorV2(),
                ])
                cls_input = cls_transform(image=after_clahe)["image"].unsqueeze(0).to(device)
                
                gradcam = GradCAM(cls_model, cls_model.vit.encoder.layers[-1].ln_1)
                cam = gradcam.generate_cam(cls_input, target_class=1)
                gradcam_overlay, gradcam_heatmap = visualize_gradcam(after_clahe, cam, 224)
                
                gradcam_save_path = os.path.join(args.output_folder, "gradcam", f"gradcam_{img_name}")
                plt.figure(figsize=(15, 5))
                plt.subplot(131), plt.imshow(after_clahe), plt.title("Enhanced Image"), plt.axis("off")
                plt.subplot(132), plt.imshow(gradcam_heatmap, cmap="jet"), plt.title("Grad-CAM Heatmap"), plt.axis("off")
                plt.subplot(133), plt.imshow(gradcam_overlay), plt.title("Grad-CAM Overlay"), plt.axis("off")
                plt.savefig(gradcam_save_path, dpi=100, bbox_inches="tight")
                plt.close()

                # Attention rollout
                with torch.no_grad():
                    _, attn_weights = cls_model(cls_input)
                rollout = attention_rollout(attn_weights)
                attn_overlay, attn_heatmap = visualize_attention_rollout(after_clahe, rollout, 224)
                
                attn_save_path = os.path.join(args.output_folder, "attention_rollout", f"attn_{img_name}")
                plt.figure(figsize=(15, 5))
                plt.subplot(131), plt.imshow(after_clahe), plt.title("Enhanced Image"), plt.axis("off")
                plt.subplot(132), plt.imshow(attn_heatmap, cmap="jet"), plt.title("Attention Heatmap"), plt.axis("off")
                plt.subplot(133), plt.imshow(attn_overlay), plt.title("Attention Overlay"), plt.axis("off")
                plt.savefig(attn_save_path, dpi=100, bbox_inches="tight")
                plt.close()
            except Exception as e:
                print(f"    ⚠️  Error with classification for {img_name}: {e}")

        # Combined visualization
        combined_save_path = os.path.join(args.output_folder, "all_stages_combined", f"combined_{img_name}")
        plt.figure(figsize=(20, 10))
        plt.subplot(241), plt.imshow(img), plt.title("1. Original"), plt.axis("off")
        plt.subplot(242), plt.imshow(seg_mask, cmap="gray"), plt.title("2. Segmentation Mask"), plt.axis("off")
        plt.subplot(243), plt.imshow(roi_img), plt.title("3. ROI Extraction"), plt.axis("off")
        plt.subplot(244), plt.imshow(enhanced_rgb), plt.title("4. RGB Enhanced"), plt.axis("off")
        plt.subplot(245), plt.imshow(blended_rgb), plt.title("5. Blended RGB"), plt.axis("off")
        plt.subplot(246), plt.imshow(after_clahe), plt.title("6. LAB-CLAHE"), plt.axis("off")
        if cls_model is not None and 'gradcam_overlay' in locals():
            plt.subplot(247), plt.imshow(gradcam_overlay), plt.title("7. Grad-CAM"), plt.axis("off")
            plt.subplot(248), plt.imshow(attn_overlay), plt.title("8. Attention Rollout"), plt.axis("off")
        plt.suptitle(f"Processing Results for {img_name}", fontsize=16)
        plt.tight_layout()
        plt.savefig(combined_save_path, dpi=150, bbox_inches="tight")
        plt.close()

    print("\n" + "=" * 80)
    print("PROCESSING COMPLETE!")
    print("=" * 80)
    print(f"Outputs saved to: {args.output_folder}")
    print("  - segmentation_masks/: Binary masks from best model")
    print("  - roi_extraction/: ROI extraction visualizations")
    print("  - rgb_enhancement/: RGB enhancement results")
    print("  - lab_clahe/: LAB-CLAHE enhancement")
    print("  - gradcam/: Grad-CAM explainability")
    print("  - attention_rollout/: Attention rollout")
    print("  - all_stages_combined/: Full pipeline visualizations")


if __name__ == "__main__":
    main()
