
"""
Training script for improved A-DenseUNet with:
- CBAM Attention
- ASPP Module
- Deep Supervision
- Residual Connections
- GroupNorm instead of BatchNorm
"""
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import albumentations as A
from albumentations.pytorch import ToTensorV2
import numpy as np
from PIL import Image
from tqdm import tqdm
from adenseunet_improved_v2 import ADenseUNet_Improved, DeepSupervisionLoss


# --------------------------
# Dataset
# --------------------------
class PolypDataset(Dataset):
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
        mask = (mask > 0.5).astype(np.float32)
        
        if self.transform:
            augmented = self.transform(image=image, mask=mask)
            image = augmented["image"]
            mask = augmented["mask"]
        
        return image, mask.unsqueeze(0)


# --------------------------
# Metrics
# --------------------------
def dice_score(logits, targets, smooth=1e-6):
    preds = torch.sigmoid(logits)
    preds = (preds > 0.5).float()
    preds = preds.view(-1)
    targets = targets.view(-1)
    intersection = (preds * targets).sum()
    return (2.0 * intersection + smooth) / (preds.sum() + targets.sum() + smooth)


# --------------------------
# Training Setup
# --------------------------
def main():
    # Config
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    IMAGE_SIZE = 224
    BATCH_SIZE = 8  # Larger batch size to better use GroupNorm
    EPOCHS = 200
    LEARNING_RATE = 1e-4
    DATA_ROOT = "archive (1)/PNG"
    CHECKPOINT_DIR = "checkpoints"
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    
    print("=" * 80)
    print("Training Improved A-DenseUNet v2")
    print("=" * 80)
    
    # Data transforms (strong augmentations)
    train_transform = A.Compose([
        A.Resize(IMAGE_SIZE, IMAGE_SIZE),
        A.RandomRotate90(p=0.8),
        A.HorizontalFlip(p=0.7),
        A.VerticalFlip(p=0.5),
        A.Affine(scale=(0.8, 1.2), translate_percent=(0.1, 0.1), rotate=(-30, 30), p=0.7),
        A.RandomBrightnessContrast(brightness_limit=0.3, contrast_limit=0.3, p=0.7),
        A.GaussianBlur(blur_limit=(3, 5), p=0.3),
        A.CLAHE(clip_limit=2.0, tile_grid_size=(8, 8), p=0.3),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2(),
    ])
    
    val_transform = A.Compose([
        A.Resize(IMAGE_SIZE, IMAGE_SIZE),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2(),
    ])
    
    # Split dataset
    full_dataset = PolypDataset(DATA_ROOT)
    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(
        full_dataset, [train_size, val_size], generator=torch.Generator().manual_seed(42)
    )
    
    train_dataset.dataset.transform = train_transform
    val_dataset.dataset.transform = val_transform
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    
    print(f"Train set: {len(train_dataset)}, Val set: {len(val_dataset)}")
    print()
    
    # Model
    model = ADenseUNet_Improved(in_channels=3, out_channels=1).to(DEVICE)
    criterion = DeepSupervisionLoss(final_weight=0.4, ds1_weight=0.2, ds2_weight=0.2, ds3_weight=0.2)
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)
    
    # Learning rate scheduler (cosine annealing)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
    
    # Training loop
    best_val_dice = 0.0
    
    for epoch in range(EPOCHS):
        model.train()
        train_loss = 0.0
        train_dice = 0.0
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS} (Train)")
        for images, masks in pbar:
            images, masks = images.to(DEVICE), masks.to(DEVICE)
            
            optimizer.zero_grad()
            final_out, ds1_out, ds2_out, ds3_out = model(images, return_deep_supervision=True)
            loss = criterion(final_out, ds1_out, ds2_out, ds3_out, masks)
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            with torch.no_grad():
                dice = dice_score(final_out, masks)
                train_loss += loss.item() * images.size(0)
                train_dice += dice.item() * images.size(0)
            
            pbar.set_postfix({"Loss": f"{loss.item():.4f}", "Dice": f"{dice.item():.4f}"})
        
        avg_train_loss = train_loss / len(train_loader.dataset)
        avg_train_dice = train_dice / len(train_loader.dataset)
        
        # Validation
        model.eval()
        val_loss = 0.0
        val_dice = 0.0
        
        with torch.no_grad():
            pbar = tqdm(val_loader, desc=f"Epoch {epoch+1}/{EPOCHS} (Val)")
            for images, masks in pbar:
                images, masks = images.to(DEVICE), masks.to(DEVICE)
                final_out, ds1_out, ds2_out, ds3_out = model(images, return_deep_supervision=True)
                loss = criterion(final_out, ds1_out, ds2_out, ds3_out, masks)
                dice = dice_score(final_out, masks)
                
                val_loss += loss.item() * images.size(0)
                val_dice += dice.item() * images.size(0)
                pbar.set_postfix({"Val Loss": f"{loss.item():.4f}", "Val Dice": f"{dice.item():.4f}"})
        
        avg_val_loss = val_loss / len(val_loader.dataset)
        avg_val_dice = val_dice / len(val_loader.dataset)
        
        # Update scheduler
        scheduler.step()
        
        print(f"\nEpoch {epoch+1}:")
        print(f"  Train - Loss: {avg_train_loss:.4f}, Dice: {avg_train_dice:.4f}")
        print(f"  Val   - Loss: {avg_val_loss:.4f}, Dice: {avg_val_dice:.4f}")
        print()
        
        # Save best model
        if avg_val_dice > best_val_dice:
            best_val_dice = avg_val_dice
            torch.save({
                "epoch": epoch+1,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "best_dice": best_val_dice
            }, os.path.join(CHECKPOINT_DIR, "best_adenseunet_improved_v2.pth"))
            print(f"NEW BEST MODEL SAVED (Dice: {best_val_dice:.4f}")
            print()
        
        # Periodic save every 50 epochs
        if (epoch+1) % 50 == 0:
            torch.save({
                "epoch": epoch+1,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "best_dice": best_val_dice
            }, os.path.join(CHECKPOINT_DIR, f"checkpoint_improved_v2_epoch_{epoch+1}.pth"))
            print(f"Periodic checkpoint saved (epoch {epoch+1})")
            print()
    
    print("=" * 80)
    print("TRAINING COMPLETE!")
    print(f"Best validation Dice score: {best_val_dice:.4f}")
    print("=" * 80)


if __name__ == "__main__":
    main()
