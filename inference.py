import torch
import numpy as np
from PIL import Image
import os
import cv2

from src.models.a_denseunet import ADenseUNet
from src.models.vit_classifier import VisionTransformer
from src.enhancement.pipeline import enhance_pipeline
from src.explainability.explainers import GradCAM, attention_rollout, visualize_cam, visualize_attention_rollout


class PolypFramework:
    def __init__(self, seg_checkpoint, cls_checkpoint, device="cpu"):
        self.device = device
        
        # Load segmentation model
        self.seg_model = ADenseUNet(in_channels=3, out_channels=1).to(device)
        if seg_checkpoint and os.path.exists(seg_checkpoint):
            checkpoint = torch.load(seg_checkpoint, map_location=device)
            self.seg_model.load_state_dict(checkpoint["model_state_dict"])
        self.seg_model.eval()
        
        # Load classification model
        self.cls_model = VisionTransformer(
            img_size=224, patch_size=16, in_channels=3, num_classes=2,
            embed_dim=768, depth=12, num_heads=12
        ).to(device)
        if cls_checkpoint and os.path.exists(cls_checkpoint):
            checkpoint = torch.load(cls_checkpoint, map_location=device)
            self.cls_model.load_state_dict(checkpoint["model_state_dict"])
        self.cls_model.eval()
        
        # Preprocessing
        self.mean = [0.0, 0.0, 0.0]
        self.std = [1.0, 1.0, 1.0]
        
    def preprocess(self, image):
        """Preprocess image for model input"""
        img = np.array(image).astype(np.float32) / 255.0
        img = (img - self.mean) / self.std
        img = np.transpose(img, (2, 0, 1))
        return torch.tensor(img).unsqueeze(0).float().to(self.device)
        
    def predict(self, image):
        """Complete pipeline: segmentation -> ROI -> enhancement -> classification"""
        # Preprocess
        input_tensor = self.preprocess(image)
        
        # Step 1: Segmentation
        with torch.no_grad():
            mask = self.seg_model(input_tensor)
            mask = mask.squeeze().cpu().numpy()
            
        # Step 2-3: ROI extraction and enhancement
        image_np = np.array(image)
        enhanced_roi = enhance_pipeline(image_np, mask)
        
        # Step 4: Classification
        enhanced_pil = Image.fromarray(enhanced_roi).resize((224, 224))
        cls_tensor = self.preprocess(enhanced_pil)
        
        with torch.no_grad():
            cls_output, attn_weights = self.cls_model(cls_tensor)
            probabilities = torch.softmax(cls_output, dim=1)
            pred_class = torch.argmax(probabilities, dim=1).item()
            
        return {
            "mask": mask,
            "enhanced_roi": enhanced_roi,
            "class": pred_class,
            "probabilities": probabilities.cpu().numpy(),
            "attention_weights": attn_weights
        }
        
    def explain(self, image):
        """Generate explanations using GradCAM and Attention Rollout"""
        input_tensor = self.preprocess(image)
        
        # GradCAM
        target_layer = self.cls_model.blocks[-1].norm1
        gradcam = GradCAM(self.cls_model, target_layer)
        cam = gradcam.generate_cam(input_tensor)
        
        # Attention Rollout
        _, attn_weights = self.cls_model(input_tensor)
        rollout = attention_rollout(attn_weights)
        
        # Visualize
        cam_vis = visualize_cam(image, cam)
        rollout_vis = visualize_attention_rollout(image, rollout)
        
        return {
            "gradcam": cam_vis,
            "attention_rollout": rollout_vis
        }


def main():
    # Example usage
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Initialize framework (update checkpoint paths)
    framework = PolypFramework(
        seg_checkpoint="checkpoints/best_segmentation_model.pth",
        cls_checkpoint="checkpoints/best_classification_model.pth",
        device=device
    )
    
    # Load image (update path)
    image_path = "example_image.jpg"
    if os.path.exists(image_path):
        image = Image.open(image_path).convert("RGB").resize((224, 224))
        
        # Predict
        result = framework.predict(image)
        print(f"Predicted class: {result['class']}")
        print(f"Probabilities: {result['probabilities']}")
        
        # Explain
        explanations = framework.explain(image)
        
        # Save results
        os.makedirs("results", exist_ok=True)
        cv2.imwrite("results/mask.png", (result['mask'] * 255).astype(np.uint8))
        cv2.imwrite("results/enhanced_roi.png", cv2.cvtColor(result['enhanced_roi'], cv2.COLOR_RGB2BGR))
        cv2.imwrite("results/gradcam.png", explanations['gradcam'])
        cv2.imwrite("results/attention_rollout.png", explanations['attention_rollout'])


if __name__ == "__main__":
    main()
