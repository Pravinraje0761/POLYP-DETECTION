
"""
Quick Boost Training Script - Fine-tunes existing best model to 93%+
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
import torch.nn.functional as F

# Import checkpoint-matching model
from adenseunet_checkpoint_matching import ADenseUNet

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
IMAGE_SIZE = 512
BATCH_SIZE = 4
LEARNING_RATE = 5e-5
EPOCHS = 200
DATA_ROOT = "archive (1)/PNG"
CHECKPOINT_DIR = "checkpoints"
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

# ==================== LOSS ====================
class DiceLoss(nn.Module):
    def __init__(self, smooth=1e-6):
        super().__init__()
        self.smooth = smooth
    def forward(self, logits, targets):
        probs = torch.sigmoid(logits)
        probs = probs.view(-1)
        targets = targets.view(-1)
        intersection = (probs * targets).sum()
        return 1 - (2. * intersection + self.smooth) / (probs.sum() + targets.sum() + self.smooth)

class TverskyLoss(nn.Module):
    def __init__(self, alpha=0.7, smooth=1e-6):
        super().__init__()
        self.alpha = alpha
        self.smooth = smooth
    def forward(self, logits, targets):
        probs = torch.sigmoid(logits)
        probs = probs.view(-1)
        targets = targets.view(-1)
        true_pos = (probs * targets).sum()
        false_neg = ((1 - probs) * targets).sum()
        false_pos = (probs * (1 - targets)).sum()
        tversky = (true_pos + self.smooth) / (true_pos + self.alpha * false_neg + (1 - self.alpha) * false_pos + self.smooth)
        return 1 - tversky

class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
    def forward(self, logits, targets):
        probs = torch.sigmoid(logits)
        probs = probs.view(-1)
        targets = targets.view(-1)
        bce = F.binary_cross_entropy(probs, targets, reduction='none')
        pt = torch.where(targets == 1, probs, 1 - probs)
        focal = self.alpha * (1 - pt) ** self.gamma * bce
        return focal.mean()

class CombinedLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.bce = nn.BCEWithLogitsLoss()
        self.dice = DiceLoss()
        self.tversky = TverskyLoss()
        self.focal = FocalLoss()
    def forward(self, logits, targets):
        return (0.1 * self.bce(logits, targets) + 0.4 * self.dice(logits, targets) + 
                0.3 * self.tversky(logits, targets) + 0.2 * self.focal(logits, targets))

# ==================== DATASET ====================
class PolypDataset(Dataset):
    def __init__(self, data_root, transform=None):
        self.data_root = data_root
        self.original_dir = os.path.join(data_root, "Original")
        self.gt_dir = os.path.join(data_root, "Ground Truth")
        self.transform = transform
        self.image_files = sorted([f for f in os.listdir(self.original_dir) if f.endswith('.png')])
    def __len__(self):
        return len(self.image_files)
    def __getitem__(self, idx):
        img_file = self.image_files[idx]
        img_path = os.path.join(self.original_dir, img_file)
        mask_path = os.path.join(self.gt_dir, img_file)
        image = np.array(Image.open(img_path).convert("RGB"))
        mask = np.array(Image.open(mask_path).convert("L"), dtype=np.float32)
        mask = (mask > 127.5).astype(np.float32)
        if self.transform:
            augmented = self.transform(image=image, mask=mask)
            image = augmented["image"]
            mask = augmented["mask"]
        return image, mask.unsqueeze(0)

def get_transforms(image_size=512, is_train=True):
    if is_train:
        return A.Compose([
            A.Resize(image_size, image_size),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.2),
            A.RandomRotate90(p=0.5),
            A.ShiftScaleRotate(shift_limit=0.15, scale_limit=0.15, rotate_limit=45, p=0.7),
            A.RandomBrightnessContrast(brightness_limit=0.25, contrast_limit=0.25, p=0.7),
            A.HueSaturationValue(hue_shift_limit=15, sat_shift_limit=30, val_shift_limit=20, p=0.5),
            A.GaussianBlur(blur_limit=(3, 5), p=0.3),
            A.CLAHE(clip_limit=4.0, tile_grid_size=(8, 8), p=0.4),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2(),
        ])
    else:
        return A.Compose([
            A.Resize(image_size, image_size),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2(),
        ])

def dice_score(logits, targets, smooth=1e-6):
    probs = torch.sigmoid(logits)
    probs = probs.view(-1)
    targets = targets.view(-1)
    intersection = (probs * targets).sum()
    return (2. * intersection + smooth) / (probs.sum() + targets.sum() + smooth)

def main():
    print("="*80)
    print("QUICK BOOST TRAINING - PUSH TO 93%+ DICE")
    print("="*80)
    
    full_dataset = PolypDataset(DATA_ROOT)
    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(
        full_dataset, [train_size, val_size], generator=torch.Generator().manual_seed(42)
    )
    train_dataset.dataset.transform = get_transforms(IMAGE_SIZE, is_train=True)
    val_dataset.dataset.transform = get_transforms(IMAGE_SIZE, is_train=False)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    print(f"Train: {len(train_dataset)}, Val: {len(val_dataset)}, Image size: {IMAGE_SIZE}")
   # Model
    model = ADenseUNet(in_channels=3, out_channels=1).to(DEVICE)
    
    # Initialize new layers with proper shapes
    model.extra_up = nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2).to(DEVICE)
    model.final_conv = nn.Conv2d(32, 1, kernel_size=1).to(DEVICE)
    
    criterion = CombinedLoss()
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=1e-6)
    best_val_dice = 0.0
    
    # Load our best checkpoint
    best_ckpt = os.path.join(CHECKPOINT_DIR, "best_adenseunet_boosted_fixed.pth")
    if os.path.exists(best_ckpt):
        checkpoint = torch.load(best_ckpt, map_location=DEVICE, weights_only=False)
        state_dict = checkpoint["model_state_dict"]
        
        # Remove the old final_conv from state dict (we'll initialize new one)
        if "final_conv.weight" in state_dict:
            del state_dict["final_conv.weight"]
        if "final_conv.bias" in state_dict:
            del state_dict["final_conv.bias"]
            
        # Load the rest
        model.load_state_dict(state_dict, strict=False)
        best_val_dice = checkpoint["best_dice"]
        print(f"Loaded checkpoint with Dice: {best_val_dice:.4f} (updated final_conv and added extra_up)")
    
    print("\nStarting training...")
    for epoch in range(EPOCHS):
        model.train()
        train_loss = 0.0
        train_dice = 0.0
        
        for images, masks in tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS} (Train)"):
            images, masks = images.to(DEVICE), masks.to(DEVICE)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, masks)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            with torch.no_grad():
                dice = dice_score(outputs, masks)
                train_loss += loss.item() * images.size(0)
                train_dice += dice.item() * images.size(0)
        
        scheduler.step()
        
        model.eval()
        val_loss = 0.0
        val_dice = 0.0
        with torch.no_grad():
            for images, masks in tqdm(val_loader, desc=f"Epoch {epoch+1}/{EPOCHS} (Val)"):
                images, masks = images.to(DEVICE), masks.to(DEVICE)
                outputs = model(images)
                loss = criterion(outputs, masks)
                dice = dice_score(outputs, masks)
                val_loss += loss.item() * images.size(0)
                val_dice += dice.item() * images.size(0)
        
        avg_train_dice = train_dice / len(train_loader.dataset)
        avg_val_dice = val_dice / len(val_loader.dataset)
        
        print(f"Epoch {epoch+1}: Train Dice={avg_train_dice:.4f}, Val Dice={avg_val_dice:.4f}")
        
        if avg_val_dice > best_val_dice:
            best_val_dice = avg_val_dice
            torch.save({
                "epoch": epoch + 1,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "best_dice": best_val_dice
            }, os.path.join(CHECKPOINT_DIR, "best_adenseunet_93_plus_final.pth"))
            print(f"[NEW BEST] Val Dice: {best_val_dice:.4f}")
            
            if best_val_dice >= 0.93:
                print("\n" + "="*80)
                print("🎉 SUCCESS! 93%+ DICE SCORE ACHIEVED! 🎉")
                print(f"Final Val Dice: {best_val_dice:.4f}")
                print("="*80)
                break
    
    print("\nTraining complete!")
    print(f"Best Val Dice: {best_val_dice:.4f}")

if __name__ == "__main__":
    main()

