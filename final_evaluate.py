
import os
import torch
import torch.nn as nn
import albumentations as A
from albumentations.pytorch import ToTensorV2
from tqdm import tqdm
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from adenseunet_boosted import ADenseUNet_Boosted


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
        
        return image, mask.unsqueeze(0), img_file, np.array(Image.open(img_path).convert("RGB"))


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    output_dir = "boosted_model_results"
    os.makedirs(output_dir, exist_ok=True)
    
    print("=" * 80)
    print("FINAL EVALUATION OF BOOSTED MODEL (90.86% DICE)")
    print("=" * 80)
    
    # Load model
    model = ADenseUNet_Boosted(in_channels=3, out_channels=1).to(device)
    checkpoint = torch.load("checkpoints/best_adenseunet_boosted.pth", map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    
    print(f"Model loaded successfully!")
    print(f"   - Best val dice: {checkpoint['best_dice']:.4f}")
    print(f"   - Trained for {checkpoint['epoch']} epochs\n")
    
    # Dataset
    transform = A.Compose([
        A.Resize(112, 112),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2(),
    ])
    dataset = PolypDataset("archive (1)/PNG", transform)
    # Use first 20 samples for visualization, evaluate all
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False)
    
    total_dice = 0.0
    total_iou = 0.0
    
    with torch.no_grad():
        for i, (img, mask, name, orig_img) in enumerate(tqdm(dataloader, desc="Evaluating")):
            img, mask = img.to(device), mask.to(device)
            logits = model(img)
            preds = torch.sigmoid(logits)
            
            dice = dice_score(preds, mask)
            iou = iou_score(preds, mask)
            
            total_dice += dice.item()
            total_iou += iou.item()
            
            # Visualize first 20 samples
            if i < 20:
                orig_img = orig_img[0].numpy()
                pred_mask = preds[0, 0].cpu().numpy()
                gt_mask = mask[0, 0].cpu().numpy()
                
                fig, axes = plt.subplots(2, 3, figsize=(15, 10))
                axes[0,0].imshow(orig_img)
                axes[0,0].set_title("Original Image")
                axes[0,0].axis("off")
                
                axes[0,1].imshow(gt_mask, cmap="gray")
                axes[0,1].set_title("Ground Truth")
                axes[0,1].axis("off")
                
                axes[0,2].imshow(orig_img)
                axes[0,2].imshow(gt_mask, cmap="jet", alpha=0.5)
                axes[0,2].set_title("GT Overlay")
                axes[0,2].axis("off")
                
                axes[1,0].imshow(pred_mask, cmap="gray")
                axes[1,0].set_title(f"Prediction (Dice: {dice.item():.4f})")
                axes[1,0].axis("off")
                
                axes[1,1].imshow(orig_img)
                axes[1,1].imshow(pred_mask, cmap="jet", alpha=0.5)
                axes[1,1].set_title("Prediction Overlay")
                axes[1,1].axis("off")
                
                axes[1,2].text(0.5, 0.5, f"Image: {name[0]}\nDice Score: {dice.item():.4f}\nIoU: {iou.item():.4f}", 
                              ha="center", va="center", fontsize=12)
                axes[1,2].axis("off")
                
                plt.tight_layout()
                plt.savefig(os.path.join(output_dir, f"result_{i+1:02d}_{name[0]}"), dpi=150, bbox_inches="tight")
                plt.close()
    
    avg_dice = total_dice / len(dataloader)
    avg_iou = total_iou / len(dataloader)
    
    print("\n" + "=" * 80)
    print("EVALUATION RESULTS")
    print("=" * 80)
    print(f"Total samples: {len(dataloader)}")
    print(f"Average Dice Score: {avg_dice:.4f}")
    print(f"Average IoU Score: {avg_iou:.4f}")
    print(f"\nVisualizations saved to: {output_dir}")
    print("=" * 80)


if __name__ == "__main__":
    main()
