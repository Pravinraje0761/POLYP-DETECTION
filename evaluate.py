"""
Stage 7 & 10: Evaluation and explainability metrics.
"""
import os
import torch
import numpy as np
from PIL import Image
import argparse
from tqdm import tqdm

from dataset import PolypDataset, get_transforms
from adenseunet_simple import ADenseUNet
from vit_classifier import PolypViT
from preprocessing import complete_preprocessing_pipeline
from roi_extraction import soft_roi_extraction
from rgb_enhancement import selective_rgb_enhancement
from clahe import lab_clahe_enhancement
from gradcam import GradCAM, visualize_gradcam
from attention_rollout import attention_rollout, visualize_attention_rollout
from metrics import (
    compute_classification_metrics, attention_concentration_ratio,
    gradcam_iou, pointing_game_accuracy
)
from visualization import (
    visualize_stage2, visualize_stage3, visualize_stage4,
    visualize_stage8, visualize_stage9
)


def parse_args():
    parser = argparse.ArgumentParser(description="Polyp Detection Evaluation")
    parser.add_argument("--data_root", type=str,
                        default="hyper-kvasir-labeled-images (1)/labeled-images",
                        help="Path to dataset root")
    parser.add_argument("--seg_checkpoint", type=str, required=True,
                        help="Path to A-DenseUNet checkpoint")
    parser.add_argument("--cls_checkpoint", type=str, required=True,
                        help="Path to ViT classifier checkpoint")
    parser.add_argument("--image_size", type=int, default=224, help="Image size")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu",
                        help="Device to use")
    parser.add_argument("--output_dir", type=str, default="evaluation_outputs",
                        help="Output directory")
    parser.add_argument("--num_samples", type=int, default=50,
                        help="Number of samples to evaluate for explainability")
    return parser.parse_args()


def main():
    args = parse_args()
    device = torch.device(args.device)
    
    # Create output directories
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, "stage2"), exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, "stage3"), exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, "stage4"), exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, "stage8"), exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, "stage9"), exist_ok=True)
    
    # Load models
    print("Loading models...")
    seg_model = ADenseUNet(in_channels=3, out_channels=1).to(device)
    seg_checkpoint = torch.load(args.seg_checkpoint, map_location=device, weights_only=False)
    seg_model.load_state_dict(seg_checkpoint["model_state_dict"])
    seg_model.eval()
    
    cls_model = PolypViT(num_classes=2, pretrained=False).to(device)
    cls_checkpoint = torch.load(args.cls_checkpoint, map_location=device, weights_only=False)
    cls_model.load_state_dict(cls_checkpoint["model_state_dict"])
    cls_model.eval()
    
    # Load a few samples for visualization
    print("Preparing data...")
    from dataset import load_hyper_kvasir_data
    image_paths, labels = load_hyper_kvasir_data(args.data_root)
    # Take first args.num_samples samples
    image_paths = image_paths[:args.num_samples]
    labels = labels[:args.num_samples]
    
    transform = get_transforms(args.image_size, is_train=False)
    dataset = PolypDataset(image_paths, labels=labels, transform=transform, image_size=args.image_size)
    
    # Initialize Grad-CAM
    # Target the last layer norm of the ViT encoder
    target_layer = cls_model.vit.encoder.layers[-1].ln_1
    gradcam = GradCAM(cls_model, target_layer)
    
    # Evaluate explainability metrics
    print("Evaluating explainability metrics...")
    acr_scores = []
    gradcam_iou_scores = []
    pointing_game_scores = []
    
    for idx in tqdm(range(len(dataset)), desc="Processing samples"):
        item = dataset[idx]
        image_tensor = item["image"].unsqueeze(0).to(device)
        label = item["label"]
        
        # Convert tensor back to numpy for preprocessing
        img_np = image_tensor.squeeze().permute(1, 2, 0).cpu().numpy()
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        img_np = std * img_np + mean
        img_np = np.clip(img_np, 0, 1) * 255
        img_np = img_np.astype(np.uint8)
        
        # Complete preprocessing pipeline
        pp = complete_preprocessing_pipeline(img_np, seg_model, device)
        
        # Stage 2: Visualize ROI extraction
        visualize_stage2(
            pp["original"], pp["prob_mask"], pp["roi"],
            save_path=os.path.join(args.output_dir, "stage2", f"sample_{idx:03d}.png")
        )
        
        # Stage 3: Visualize RGB enhancement
        visualize_stage3(
            pp["original"], pp["enhanced_rgb"], pp["blended_rgb"],
            save_path=os.path.join(args.output_dir, "stage3", f"sample_{idx:03d}.png")
        )
        
        # Stage 4: Visualize CLAHE
        visualize_stage4(
            pp["before_clahe"], pp["after_clahe"],
            save_path=os.path.join(args.output_dir, "stage4", f"sample_{idx:03d}.png")
        )
        
        # Process enhanced image for classification
        from albumentations import Compose, Resize, Normalize
        from albumentations.pytorch import ToTensorV2
        transform_cls = Compose([
            Resize(args.image_size, args.image_size),
            Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2()
        ])
        processed_tensor = transform_cls(image=pp["after_clahe"])["image"].unsqueeze(0).to(device)
        
        # Stage 8: Grad-CAM
        cam = gradcam.generate_cam(processed_tensor, target_class=label)
        gradcam_overlay, gradcam_heatmap = visualize_gradcam(pp["after_clahe"], cam, args.image_size)
        
        # Stage 9: Attention Rollout
        with torch.no_grad():
            _, attn_weights = cls_model(processed_tensor)
        rollout = attention_rollout(attn_weights)
        attn_overlay, attn_heatmap = visualize_attention_rollout(pp["after_clahe"], rollout, args.image_size)
        
        # Visualize stage 8
        visualize_stage8(
            pp["original"], pp["prob_mask"], pp["roi"], pp["after_clahe"],
            gradcam_overlay, gradcam_heatmap,
            save_path=os.path.join(args.output_dir, "stage8", f"sample_{idx:03d}.png")
        )
        
        # Visualize stage 9
        visualize_stage9(
            gradcam_overlay, attn_overlay, attn_heatmap,
            save_path=os.path.join(args.output_dir, "stage9", f"sample_{idx:03d}.png")
        )
        
        # Stage 10: Compute explainability metrics
        # For ACR and IoU, we need a mask - we'll use our segmentation mask
        mask = pp["prob_mask"]
        
        # ACR for Grad-CAM
        acr_gradcam = attention_concentration_ratio(cam, mask)
        acr_scores.append(acr_gradcam)
        
        # Grad-CAM IoU
        gc_iou = gradcam_iou(cam, mask)
        gradcam_iou_scores.append(gc_iou)
        
        # Pointing Game
        pg_acc = pointing_game_accuracy(cam, mask)
        pointing_game_scores.append(pg_acc)
    
    # Print explainability metrics
    print("\nExplainability Metrics:")
    print(f"  Mean ACR: {np.mean(acr_scores):.4f} ± {np.std(acr_scores):.4f}")
    print(f"  Mean Grad-CAM IoU: {np.mean(gradcam_iou_scores):.4f} ± {np.std(gradcam_iou_scores):.4f}")
    print(f"  Pointing Game Accuracy: {np.mean(pointing_game_scores):.4f}")
    
    # Save metrics
    metrics = {
        "acr": acr_scores,
        "gradcam_iou": gradcam_iou_scores,
        "pointing_game": pointing_game_scores
    }
    np.save(os.path.join(args.output_dir, "explainability_metrics.npy"), metrics)
    
    print(f"\nEvaluation complete! Results saved to {args.output_dir}")


if __name__ == "__main__":
    main()
