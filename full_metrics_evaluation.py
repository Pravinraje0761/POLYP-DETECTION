
import os
import torch
import torch.nn as nn
import albumentations as A
from albumentations.pytorch import ToTensorV2
from tqdm import tqdm
from PIL import Image
import numpy as np
from adenseunet_simple import ADenseUNet


# ==================== METRICS CALCULATION ====================
def calculate_metrics(pred_mask, gt_mask, threshold=0.5):
    """
    Calculate all metrics for a single image
    """
    # Threshold predictions
    pred_binary = (pred_mask > threshold).astype(np.float32)
    gt_binary = gt_mask.astype(np.float32)
    
    # Calculate TP, TN, FP, FN
    tp = np.sum((pred_binary == 1) & (gt_binary == 1))
    tn = np.sum((pred_binary == 0) & (gt_binary == 0))
    fp = np.sum((pred_binary == 1) & (gt_binary == 0))
    fn = np.sum((pred_binary == 0) & (gt_binary == 1))
    
    # Calculate metrics
    smooth = 1e-6
    
    pixel_accuracy = (tp + tn) / (tp + tn + fp + fn + smooth)
    dice = (2 * tp + smooth) / (2 * tp + fp + fn + smooth)
    iou = (tp + smooth) / (tp + fp + fn + smooth)
    precision = (tp + smooth) / (tp + fp + smooth)
    recall = (tp + smooth) / (tp + fn + smooth)
    specificity = (tn + smooth) / (tn + fp + smooth)
    
    return {
        'pixel_accuracy': pixel_accuracy,
        'dice': dice,
        'iou': iou,
        'precision': precision,
        'recall': recall,
        'specificity': specificity
    }


# ==================== DATASET ====================
class PolypDataset(torch.utils.data.Dataset):
    def __init__(self, data_root, transform=None):
        self.data_root = data_root
        self.original_dir = os.path.join(data_root, "Original")
        self.gt_dir = os.path.join(data_root, "Ground Truth")
        self.transform = transform
        self.image_files = sorted([f for f in os.listdir(self.original_dir) if f.endswith(".png")])
    
    def __len__(self):
        return len(self.image_files)
    
    def __getitem__(self, idx):
        img_file = self.image_files[idx]
        img_path = os.path.join(self.original_dir, img_file)
        gt_path = os.path.join(self.gt_dir, img_file)
        
        image = np.array(Image.open(img_path).convert("RGB"))
        mask = np.array(Image.open(gt_path).convert("L"), dtype=np.float32) / 255.0
        
        if self.transform:
            augmented = self.transform(image=image, mask=mask)
            image = augmented["image"]
            mask = augmented["mask"]
        
        return image, mask.unsqueeze(0), img_file


# ==================== MAIN EVALUATION ====================
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    print("=" * 80)
    print("COMPREHENSIVE METRICS EVALUATION")
    print("=" * 80)
    
    # Dataset info
    DATA_ROOT = "archive (1)/PNG"
    DATASET_NAME = "Archive (1) Polyp Dataset"
    
    # Load model
    model = ADenseUNet(in_channels=3, out_channels=1).to(device)
    checkpoint_path = "checkpoints/best_adenseunet_improved.pth"
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    
    print(f"\nModel loaded: {checkpoint_path}")
    print(f"Best Val Dice: {checkpoint['best_dice']:.4f}")
    print(f"Trained Epochs: {checkpoint['epoch']}")
    
    # Dataset
    transform = A.Compose([
        A.Resize(224, 224),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2(),
    ])
    dataset = PolypDataset(DATA_ROOT, transform)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False)
    
    total_images = len(dataset)
    
    # Initialize metric accumulators
    total_metrics = {
        'pixel_accuracy': 0.0,
        'dice': 0.0,
        'iou': 0.0,
        'precision': 0.0,
        'recall': 0.0,
        'specificity': 0.0
    }
    
    # Evaluate all images
    print(f"\nEvaluating {total_images} images...")
    with torch.no_grad():
        for img, mask, name in tqdm(dataloader, desc="Evaluating"):
            img, mask = img.to(device), mask.to(device)
            
            # Predict
            logits = model(img)
            pred_mask = torch.sigmoid(logits)[0, 0].cpu().numpy()
            gt_mask = mask[0, 0].cpu().numpy()
            
            # Calculate metrics
            metrics = calculate_metrics(pred_mask, gt_mask)
            
            # Accumulate
            for key in total_metrics:
                total_metrics[key] += metrics[key]
    
    # Calculate averages
    avg_metrics = {}
    for key in total_metrics:
        avg_metrics[key] = total_metrics[key] / total_images
    
    # Print results
    print("\n" + "=" * 80)
    print("FINAL EVALUATION RESULTS")
    print("=" * 80)
    print(f"Dataset: {DATASET_NAME}")
    print(f"Number of images: {total_images}")
    print("\nMetrics:")
    print(f"  Pixel Accuracy: {avg_metrics['pixel_accuracy']:.4f} ({avg_metrics['pixel_accuracy']*100:.2f}%)")
    print(f"  Dice: {avg_metrics['dice']:.4f} ({avg_metrics['dice']*100:.2f}%)")
    print(f"  IoU: {avg_metrics['iou']:.4f} ({avg_metrics['iou']*100:.2f}%)")
    print(f"  Precision: {avg_metrics['precision']:.4f} ({avg_metrics['precision']*100:.2f}%)")
    print(f"  Recall: {avg_metrics['recall']:.4f} ({avg_metrics['recall']*100:.2f}%)")
    print(f"  Specificity: {avg_metrics['specificity']:.4f} ({avg_metrics['specificity']*100:.2f}%)")
    print("=" * 80)


if __name__ == "__main__":
    main()
