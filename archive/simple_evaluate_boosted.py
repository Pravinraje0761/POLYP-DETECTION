
import os
import torch
import torch.nn as nn
import albumentations as A
from albumentations.pytorch import ToTensorV2
from tqdm import tqdm
from PIL import Image
import numpy as np
from adenseunet_simple_original import ADenseUNet


def dice_score(preds, targets, smooth=1e-6):
    preds = preds.view(-1)
    targets = targets.view(-1)
    intersection = (preds * targets).sum()
    return (2. * intersection + smooth) / (preds.sum() + targets.sum() + smooth)


def iou_score(preds, targets, smooth=1e-6):
    preds = preds.view(-1)
    targets = targets.view(-1)
    intersection = (preds * targets).sum()
    union = preds.sum() + targets.sum() - intersection
    return (intersection + smooth) / (union + smooth)


class PolypDataset(torch.utils.data.Dataset):
    def __init__(self, root, transform=None):
        self.root = root
        self.transform = transform
        self.original_dir = os.path.join(root, "Original")
        self.gt_dir = os.path.join(root, "Ground Truth")
        self.image_files = sorted([f for f in os.listdir(self.original_dir) if f.endswith(".png")])
    
    def __len__(self):
        return len(self.image_files)
    
    def __getitem__(self, idx):
        img_path = os.path.join(self.original_dir, self.image_files[idx])
        gt_path = os.path.join(self.gt_dir, self.image_files[idx])
        
        img = np.array(Image.open(img_path).convert("RGB"))
        mask = np.array(Image.open(gt_path).convert("L"), dtype=np.float32) / 255.0
        
        if self.transform:
            augmented = self.transform(image=img, mask=mask)
            img = augmented["image"]
            mask = augmented["mask"]
        
        return img, mask.unsqueeze(0), self.image_files[idx]


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("=" * 70)
    print("EVALUATING BOOSTED MODEL - 90.86% DICE SCORE")
    print("=" * 70)
    
    # Load model
    model = ADenseUNet(in_channels=3, out_channels=1).to(device)
    checkpoint = torch.load("checkpoints/best_adenseunet_boosted.pth", map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    print(f"Loaded checkpoint - Best Dice: {checkpoint['best_dice']:.4f}, Epoch: {checkpoint['epoch']}\n")
    
    # Create dataset
    transform = A.Compose([
        A.Resize(224, 224),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2(),
    ])
    dataset = PolypDataset("archive (1)/PNG", transform)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False)
    
    total_dice = 0.0
    total_iou = 0.0
    
    with torch.no_grad():
        for img, mask, name in tqdm(dataloader, desc="Evaluating"):
            img, mask = img.to(device), mask.to(device)
            logits = model(img)
            preds = torch.sigmoid(logits)
            
            dice = dice_score(preds, mask)
            iou = iou_score(preds, mask)
            
            total_dice += dice.item()
            total_iou += iou.item()
    
    avg_dice = total_dice / len(dataloader)
    avg_iou = total_iou / len(dataloader)
    
    print("\n" + "=" * 70)
    print("FINAL RESULTS")
    print("=" * 70)
    print(f"Total samples evaluated: {len(dataloader)}")
    print(f"Average Dice Score: {avg_dice:.4f}")
    print(f"Average IoU Score: {avg_iou:.4f}")
    print("=" * 70)


if __name__ == "__main__":
    main()
