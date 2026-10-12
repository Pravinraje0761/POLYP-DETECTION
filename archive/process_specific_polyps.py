
import os
import numpy as np
from PIL import Image
import cv2
from polypgate.preprocessing.pipeline import complete_preprocessing_pipeline
import torch
import albumentations as A
from albumentations.pytorch import ToTensorV2
from adenseunet_simple import ADenseUNet
from polypgate.models.vit_classifier import PolypViT
from polypgate.visualization import visualize_stage2, visualize_stage3, visualize_stage4
from polypgate.explain.gradcam import GradCAM, visualize_gradcam
from polypgate.explain.attention_rollout import attention_rollout, visualize_attention_rollout

def main():
    # Path to the specific polyp folder
    polyp_folder = r"C:\Users\ADMIN\Desktop\POLYP_DETECTION\hyper-kvasir-labeled-images (1)\labeled-images\lower-gi-tract\pathological-findings\polyps"
    
    # Create output directory with subfolders for each stage
    output_base = "specific_polyp_results"
    os.makedirs(output_base, exist_ok=True)
    os.makedirs(os.path.join(output_base, "stage2"), exist_ok=True)
    os.makedirs(os.path.join(output_base, "stage3"), exist_ok=True)
    os.makedirs(os.path.join(output_base, "stage4"), exist_ok=True)
    os.makedirs(os.path.join(output_base, "stage8"), exist_ok=True)
    os.makedirs(os.path.join(output_base, "stage9"), exist_ok=True)
    
    # Load models
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    seg_model = ADenseUNet(in_channels=3, out_channels=1).to(device)
    seg_checkpoint = torch.load("checkpoints/best_adenseunet.pth", map_location=device, weights_only=False)
    seg_model.load_state_dict(seg_checkpoint["model_state_dict"])
    seg_model.eval()
    
    cls_model = PolypViT(num_classes=2, pretrained=False).to(device)
    cls_checkpoint = torch.load("outputs/checkpoints/best_model.pth", map_location=device, weights_only=False)
    cls_model.load_state_dict(cls_checkpoint["model_state_dict"])
    cls_model.eval()
    
    # Set up GradCAM
    target_layer = cls_model.vit.encoder.layers[-1].ln_1
    gradcam = GradCAM(model=cls_model, target_layer=target_layer)
    
    # Get first 10 images
    image_files = [f for f in os.listdir(polyp_folder) if f.endswith('.jpg')][:10]
    
    for i, img_file in enumerate(image_files):
        print(f"Processing image {i+1}/{len(image_files)}: {img_file}")
        img_path = os.path.join(polyp_folder, img_file)
        img = np.array(Image.open(img_path))
        
        # Run full preprocessing
        pp = complete_preprocessing_pipeline(img, seg_model, device)
        
        # Visualize Stage 2
        visualize_stage2(
            img,
            pp["prob_mask"],
            pp["roi"],
            os.path.join(output_base, "stage2", f"polyp_{i+1:03d}.png")
        )
        
        # Visualize Stage3
        visualize_stage3(
            pp["roi"],
            pp["enhanced_rgb"],
            pp["blended_rgb"],
            os.path.join(output_base, "stage3", f"polyp_{i+1:03d}.png")
        )
        
        # Visualize Stage4
        visualize_stage4(
            pp["blended_rgb"],
            pp["after_clahe"],
            os.path.join(output_base, "stage4", f"polyp_{i+1:03d}.png")
        )
        
        # Prepare for GradCAM and attention rollout
        transform = A.Compose([
            A.Resize(224, 224),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2()
        ])
        processed_tensor = transform(image=pp["after_clahe"])["image"].unsqueeze(0).to(device)
        
        # Resize after_clahe for GradCAM and attention visualization
        img_224 = cv2.resize(pp["after_clahe"], (224, 224))
        
        # GradCAM (Stage8)
        class_idx = 1  # Polyps are class 1
        heatmap = gradcam.generate_cam(processed_tensor, class_idx)
        gradcam_overlay, gradcam_raw = visualize_gradcam(img_224, heatmap)
        
        # Attention Rollout (Stage9)
        with torch.no_grad():
            _, attn_weights = cls_model(processed_tensor)
        
        rollout = attention_rollout(attn_weights)
        attn_overlay, attn_heatmap = visualize_attention_rollout(img_224, rollout)
        
        # Save Stage8 and 9
        import matplotlib.pyplot as plt
        
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        axes[0,0].imshow(img); axes[0,0].set_title("1. Original Image"); axes[0,0].axis('off')
        axes[0,1].imshow(pp["prob_mask"], cmap='gray'); axes[0,1].set_title("2. Prob Mask"); axes[0,1].axis('off')
        axes[0,2].imshow(pp["roi"]); axes[0,2].set_title("3. ROI"); axes[0,2].axis('off')
        axes[1,0].imshow(pp["after_clahe"]); axes[1,0].set_title("4. Final Preprocessed"); axes[1,0].axis('off')
        axes[1,1].imshow(cv2.cvtColor(gradcam_overlay, cv2.COLOR_BGR2RGB)); axes[1,1].set_title("5. GradCAM"); axes[1,1].axis('off')
        axes[1,2].imshow(cv2.cvtColor(attn_overlay, cv2.COLOR_BGR2RGB)); axes[1,2].set_title("6. Attention Rollout"); axes[1,2].axis('off')
        plt.tight_layout()
        plt.savefig(os.path.join(output_base, f"polyp_{i+1:03d}_all_stages.png"), dpi=150, bbox_inches='tight')
        plt.close()
    
    print(f"All results saved to {output_base}!")

if __name__ == "__main__":
    main()
