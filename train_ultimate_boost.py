
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
import albumentations as A
from albumentations.pytorch import ToTensorV2
import numpy as np
from PIL import Image
from tqdm import tqdm
import torch.nn.functional as F

from adenseunet_simple_original import ADenseUNet
from torch.cuda.amp import GradScaler, autocast

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
IMAGE_SIZE = 256
BATCH_SIZE = 16
LEARNING_RATE = 3e-4
EPOCHS = 500
DATA_ROOT = "combined_dataset"
CHECKPOINT_DIR = "checkpoints"
TARGET_DICE = 0.95
os.makedirs(CHECKPOINT_DIR, exist_ok=True)


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


class CombinedLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.bce = nn.BCEWithLogitsLoss()
        self.dice = DiceLoss()

    def forward(self, logits, targets):
        return 0.5 * self.bce(logits, targets) + 0.5 * self.dice(logits, targets)


class PolypDataset(Dataset):
    def __init__(self, images_dir, masks_dir, transform=None):
        self.images_dir = images_dir
        self.masks_dir = masks_dir
        self.transform = transform
        self.filenames = sorted([f for f in os.listdir(images_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))])

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        filename = self.filenames[idx]
        img_path = os.path.join(self.images_dir, filename)
        mask_path = os.path.join(self.masks_dir, filename)
        image = np.array(Image.open(img_path).convert("RGB"))
        mask = np.array(Image.open(mask_path).convert("L"), dtype=np.float32)
        mask = (mask > 127.5).astype(np.float32)
        
        if self.transform:
            augmented = self.transform(image=image, mask=mask)
            image = augmented["image"]
            mask = augmented["mask"]
        
        return image, mask.unsqueeze(0)


def get_transforms(is_train=True):
    if is_train:
        return A.Compose([
            A.Resize(IMAGE_SIZE, IMAGE_SIZE),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.RandomRotate90(p=0.5),
            A.Affine(scale=(0.7, 1.3), translate_percent=(0.15, 0.15), rotate=(-45, 45), shear=(-10, 10), p=0.8),
            A.RandomBrightnessContrast(brightness_limit=0.3, contrast_limit=0.3, p=0.8),
            A.CLAHE(clip_limit=4.0, tile_grid_size=(8, 8), p=0.5),
            A.GaussianBlur(blur_limit=(3, 7), p=0.3),
            A.ElasticTransform(alpha=120, sigma=120 * 0.05, p=0.3),
            A.GridDistortion(p=0.3),
            A.OpticalDistortion(distort_limit=0.05, p=0.3),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2()
        ])
    else:
        return A.Compose([
            A.Resize(IMAGE_SIZE, IMAGE_SIZE),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2()
        ])


def dice_score(logits, targets, smooth=1e-6):
    probs = torch.sigmoid(logits)
    probs = probs.view(-1)
    targets = targets.view(-1)
    intersection = (probs * targets).sum()
    return (2. * intersection + smooth) / (probs.sum() + targets.sum() + smooth)


def main():
    print("=" * 80)
    print("ULTIMATE SEGMENTATION BOOST - TARGET 95% DICE")
    print("=" * 80)
    
    images_dir = os.path.join(DATA_ROOT, "images")
    masks_dir = os.path.join(DATA_ROOT, "masks")
    full_dataset = PolypDataset(images_dir, masks_dir)
    
    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size], generator=torch.Generator().manual_seed(42))
    train_dataset.dataset.transform = get_transforms(is_train=True)
    val_dataset.dataset.transform = get_transforms(is_train=False)
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    
    print(f"Train: {len(train_dataset)}, Val: {len(val_dataset)}, Total: {len(full_dataset)}, Image Size: {IMAGE_SIZE}")
    
    model = ADenseUNet(in_channels=3, out_channels=1).to(DEVICE)
    model.extra_up = nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2).to(DEVICE)
    model.final_conv = nn.Conv2d(32, 1, kernel_size=1).to(DEVICE)
    
    best_val_dice = 0.0
    start_epoch = 0
    try:
        checkpoint = torch.load(os.path.join(CHECKPOINT_DIR, "best_adenseunet_boosted_fixed.pth"), map_location=DEVICE, weights_only=False)
        state_dict = checkpoint["model_state_dict"]
        if "final_conv.weight" in state_dict:
            del state_dict["final_conv.weight"]
        if "final_conv.bias" in state_dict:
            del state_dict["final_conv.bias"]
        model.load_state_dict(state_dict, strict=False)
        best_val_dice = checkpoint["best_dice"]
        print(f"Loaded checkpoint with Dice: {best_val_dice:.4f}")
    except Exception as e:
        print(f"Starting fresh training: {e}")
    
    criterion = CombinedLoss()
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=50, T_mult=2, eta_min=1e-6)
    scaler = GradScaler(enabled=torch.cuda.is_available())
    
    for epoch in range(start_epoch, EPOCHS):
        model.train()
        train_loss = 0.0
        train_dice = 0.0
        
        for images, masks in tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS} (Train)"):
            images, masks = images.to(DEVICE), masks.to(DEVICE)
            optimizer.zero_grad()
            
            if torch.cuda.is_available():
                with autocast():
                    outputs = model(images)
                    loss = criterion(outputs, masks)
                scaler.scale(loss).backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
            else:
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
        
        print(f"\nEpoch {epoch+1} - Train Dice: {avg_train_dice:.4f}, Val Dice: {avg_val_dice:.4f}")
        
        if avg_val_dice > best_val_dice:
            best_val_dice = avg_val_dice
            torch.save({
                "epoch": epoch + 1,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "best_dice": best_val_dice
            }, os.path.join(CHECKPOINT_DIR, "best_adenseunet_ultimate_boost.pth"))
            print(f"NEW BEST CHECKPOINT SAVED - Val Dice: {best_val_dice:.4f}")
            
            if best_val_dice >= TARGET_DICE:
                print("\n" + "=" * 80)
                print(f"SUCCESS! TARGET REACHED: {best_val_dice:.4f} DICE!")
                print("=" * 80)
                break
    
    print(f"\nTRAINING COMPLETE! BEST VAL DICE: {best_val_dice:.4f}")


if __name__ == "__main__":
    main()
