
"""
Full evaluation script for Stage1 model
Calculates Dice, IoU, Precision, Recall on validation set
"""
import os
import torch
import numpy as np
from PIL import Image
import albumentations as A
from albumentations.pytorch import ToTensorV2
from adenseunet_simple import ADenseUNet
from metrics import dice_score, iou_score
from tqdm import tqdm

# Configuration
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
IMAGE_SIZE = 224
CHECKPOINT_PATH = "checkpoints/best_adenseunet.pth"
DATA_ROOT = "archive (1)/PNG"
VAL_SPLIT = 0.2

# Load model
print("Loading model...")
model = ADenseUNet(in_channels=3, out_channels=1).to(DEVICE)
checkpoint = torch.load(CHECKPOINT_PATH, map_location=DEVICE)
model.load_state_dict(checkpoint["model_state_dict"])
model.eval()

# Image transforms
transform = A.Compose([
    A.Resize(IMAGE_SIZE, IMAGE_SIZE),
    A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ToTensorV2()
])

# Get dataset
original_dir = os.path.join(DATA_ROOT, "Original")
gt_dir = os.path.join(DATA_ROOT, "Ground Truth")
all_files = sorted(os.listdir(original_dir))

# Split into train/val (same as training)
np.random.seed(42)
np.random.shuffle(all_files)
split_idx = int(VAL_SPLIT * len(all_files))
val_files = all_files[:split_idx]
print(f"Evaluating on {len(val_files)} validation images...")

# Metrics
total_dice = 0.0
total_iou = 0.0
total_precision = 0.0
total_recall = 0.0

# Evaluate
with torch.no_grad():
    for img_file in tqdm(val_files, desc="Evaluating"):
        # Load image and mask
        img_path = os.path.join(original_dir, img_file)
        gt_path = os.path.join(gt_dir, img_file)
        original_img = np.array(Image.open(img_path).convert("RGB"))
        original_gt = np.array(Image.open(gt_path).convert("L"), dtype=np.float32)
        original_gt = (original_gt / 255.0) > 0.5  # Binary
        
        # Prepare tensor
        augmented = transform(image=original_img)
        img_tensor = augmented["image"].unsqueeze(0).to(DEVICE)
        
        # Predict
        outputs = model(img_tensor)
        pred_mask = torch.sigmoid(outputs).squeeze(0).squeeze(0).cpu().numpy()
        pred_mask = pred_mask > 0.5  # Binary prediction
        
        # Resize prediction to match GT size
        pred_mask_resized = np.array(Image.fromarray(pred_mask.astype(np.uint8)).resize(
            original_gt.shape[1::-1], Image.Resampling.NEAREST
        )) > 0
        
        # Convert to tensors for metric calculation
        gt_tensor = torch.tensor(original_gt.astype(np.float32)).unsqueeze(0).unsqueeze(0)
        pred_tensor = torch.tensor(pred_mask_resized.astype(np.float32)).unsqueeze(0).unsqueeze(0)
        
        # Calculate metrics
        dice = dice_score(pred_tensor, gt_tensor)
        iou = iou_score(pred_tensor, gt_tensor)
        
        # Precision and recall
        intersection = np.logical_and(pred_mask_resized, original_gt).sum()
        pred_pos = pred_mask_resized.sum()
        gt_pos = original_gt.sum()
        
        precision = intersection / pred_pos if pred_pos > 0 else 0.0
        recall = intersection / gt_pos if gt_pos > 0 else 0.0
        
        total_dice += dice.item()
        total_iou += iou.item()
        total_precision += precision
        total_recall += recall

# Calculate average metrics
avg_dice = total_dice / len(val_files)
avg_iou = total_iou / len(val_files)
avg_precision = total_precision / len(val_files)
avg_recall = total_recall / len(val_files)

# Print results
print("\n" + "="*50)
print("VALIDATION RESULTS")
print("="*50)
print(f"Average Dice Score:  {avg_dice:.4f}")
print(f"Average IoU Score:   {avg_iou:.4f}")
print(f"Average Precision:   {avg_precision:.4f}")
print(f"Average Recall:      {avg_recall:.4f}")
print(f"Epochs trained:      {checkpoint['epoch']}")
print(f"Best Dice Score:     {checkpoint['best_dice']:.4f}")
print("="*50)
