
import torch
import numpy as np
from tqdm import tqdm
from polypgate.data.dataset import get_dataloaders
from adenseunet_simple import ADenseUNet
from polypgate.models.vit_classifier import PolypViT
from polypgate.preprocessing.pipeline import complete_preprocessing_pipeline
from polypgate.metrics import compute_classification_metrics, plot_confusion_matrix, plot_roc_curve
import os

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load data
    data_root = "hyper-kvasir-labeled-images (1)/labeled-images"
    _, _, test_loader = get_dataloaders(data_root, batch_size=8)
    
    # Load models
    seg_model = ADenseUNet(in_channels=3, out_channels=1).to(device)
    seg_checkpoint = torch.load("checkpoints/best_adenseunet.pth", map_location=device, weights_only=False)
    seg_model.load_state_dict(seg_checkpoint["model_state_dict"])
    seg_model.eval()
    
    cls_model = PolypViT(num_classes=2, pretrained=False).to(device)
    cls_checkpoint = torch.load("outputs/checkpoints/best_model.pth", map_location=device, weights_only=False)
    cls_model.load_state_dict(cls_checkpoint["model_state_dict"])
    cls_model.eval()
    
    # Run evaluation
    print("Running evaluation on test set...")
    test_preds = []
    test_labels = []
    test_probs = []
    
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Testing"):
            images = batch["image"].to(device)
            labels = batch["label"].to(device)
            
            processed_images = []
            for img in images:
                img_np = img.permute(1, 2, 0).cpu().numpy()
                mean = np.array([0.485, 0.456, 0.406])
                std = np.array([0.229, 0.224, 0.225])
                img_np = std * img_np + mean
                img_np = np.clip(img_np, 0, 1) * 255
                img_np = img_np.astype(np.uint8)
                
                pp = complete_preprocessing_pipeline(img_np, seg_model, device)
                processed = pp["after_clahe"]
                
                from albumentations import Compose, Resize, Normalize
                from albumentations.pytorch import ToTensorV2
                transform = Compose([
                    Resize(224, 224),
                    Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                    ToTensorV2()
                ])
                processed_tensor = transform(image=processed)["image"]
                processed_images.append(processed_tensor)
            
            processed_images = torch.stack(processed_images).to(device)
            logits, _ = cls_model(processed_images)
            probs = torch.softmax(logits, dim=1)
            preds = torch.argmax(probs, dim=1)
            
            test_preds.extend(preds.detach().cpu().numpy())
            test_labels.extend(labels.detach().cpu().numpy())
            test_probs.extend(probs[:, 1].detach().cpu().numpy())
    
    # Compute metrics
    test_metrics = compute_classification_metrics(
        np.array(test_labels), np.array(test_preds), np.array(test_probs)
    )
    
    # Print results
    print("\nTest Set Results:")
    for k, v in test_metrics.items():
        if k != "confusion_matrix":
            print(f"  {k}: {v:.4f}")
    
    # Save plots
    os.makedirs("outputs/figures", exist_ok=True)
    plot_confusion_matrix(
        test_metrics["confusion_matrix"],
        save_path="outputs/figures/confusion_matrix.png"
    )
    plot_roc_curve(
        np.array(test_labels), np.array(test_probs),
        save_path="outputs/figures/roc_curve.png"
    )
    print("\nPlots saved to outputs/figures/")

if __name__ == "__main__":
    main()
