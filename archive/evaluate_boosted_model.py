
import os
import torch
import torch.nn as nn
import albumentations as A
from albumentations.pytorch import ToTensorV2
from tqdm import tqdm
from PIL import Image
import numpy as np
from adenseunet_simple import ADenseUNet


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


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("=" * 80)
    print("EVALUATING BEST BOOTED MODEL (90.86% DICE)")
    print("=" * 80)
    
    # Create model from adenseunet_simple
    model = ADenseUNet(in_channels=3, out_channels=1).to(device)
    
    # Load boosted checkpoint and fix keys
    checkpoint = torch.load("checkpoints/best_adenseunet_boosted.pth", map_location=device, weights_only=False)
    state_dict = checkpoint["model_state_dict"]
    new_state_dict = {}
    for k, v in state_dict.items():
        if k == "up_final.weight":
            new_state_dict["final_up.weight"] = v
        elif k == "up_final.bias":
            new_state_dict["final_up.bias"] = v
        else:
            new_state_dict[k] = v
    
    model.load_state_dict(new_state_dict)
    model.eval()
    
    print(f"✅ Loaded boosted model successfully!")
    print(f"   - Best val dice: {checkpoint['best_dice']:.4f}")
    print(f"   - Trained for {checkpoint['epoch']} epochs\n")
    
    # Evaluate on full dataset
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
    
    print("\n" + "=" * 80)
    print("📊 EVALUATION RESULTS")
    print("=" * 80)
    print(f"Total samples: {len(dataloader)}")
    print(f"✅ Average Dice Score: {avg_dice:.4f}")
    print(f"✅ Average IoU Score: {avg_iou:.4f}")
    print("=" * 80)


if __name__ == "__main__":
    main()
