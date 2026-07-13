
"""
Boost existing ADenseUNet (best_adenseunet_ultimate.pth) to 95%+ Dice.
Features:
- 512x512 image size
- Combined loss (BCE + Dice + Tversky + Focal)
- Advanced augmentations
- Better optimizer/scheduler
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

# Import our original working models
from adenseunet_simple import ADenseUNet, CombinedLoss
from torch.cuda.amp import GradScaler, autocast

# Configuration
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
IMAGE_SIZE = 512
BATCH_SIZE = 4
LEARNING_RATE = 1e-4
EPOCHS = 300
DATA_ROOT = "archive (1)/PNG"
CHECKPOINT_DIR = "checkpoints"
TARGET_DICE = 0.93  # Target: >93% Dice

os.makedirs(CHECKPOINT_DIR, exist_ok=True)


# ==================== LOSS FUNCTIONS ====================
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
        tversky = (true_pos + self.smooth) / (
            true_pos + self.alpha * false_neg + (1 - self.alpha) * false_pos + self.smooth
        )
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


class UltimateLoss(nn.Module):
    def __init__(
        self,
        bce_weight=0.2,
        dice_weight=0.3,
        tversky_weight=0.3,
        focal_weight=0.2
    ):
        super().__init__()
        self.bce = nn.BCEWithLogitsLoss()
        self.dice = DiceLoss()
        self.tversky = TverskyLoss()
        self.focal = FocalLoss()
        self.bce_weight = bce_weight
        self.dice_weight = dice_weight
        self.tversky_weight = tversky_weight
        self.focal_weight = focal_weight

    def forward(self, logits, targets):
        loss = (
            self.bce_weight * self.bce(logits, targets) +
            self.dice_weight * self.dice(logits, targets) +
            self.tversky_weight * self.tversky(logits, targets) +
            self.focal_weight * self.focal(logits, targets)
        )
        return loss


# ==================== DATASET ====================
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
        mask_path = os.path.join(self.gt_dir, img_file)
        
        image = np.array(Image.open(img_path).convert("RGB"))
        mask = np.array(Image.open(mask_path).convert("L"), dtype=np.float32)
        mask = (mask > 127.5).astype(np.float32)  # Binarize

        if self.transform:
            augmented = self.transform(image=image, mask=mask)
            image = augmented["image"]
            mask = augmented["mask"]

        return image, mask.unsqueeze(0)


# ==================== ADVANCED AUGMENTATIONS ====================
def get_transforms(image_size=512, is_train=True):
    if is_train:
        return A.Compose([
            A.Resize(height=image_size, width=image_size),
            A.Affine(scale=(0.85, 1.15), translate_percent=(0.15, 0.15), rotate=(-45, 45), p=0.7),
            A.RandomRotate90(p=0.7),
            A.HorizontalFlip(p=0.7),
            A.VerticalFlip(p=0.3),
            A.RandomBrightnessContrast(brightness_limit=0.3, contrast_limit=0.3, p=0.7),
            A.GaussianBlur(blur_limit=(3, 7), p=0.3),
            A.ElasticTransform(p=0.3, alpha=180, sigma=180 * 0.05),
            A.GridDistortion(p=0.3),
            A.OpticalDistortion(p=0.3, distort_limit=0.05),
            A.HueSaturationValue(hue_shift_limit=15, sat_shift_limit=30, val_shift_limit=20, p=0.4),
            A.CLAHE(clip_limit=4.0, tile_grid_size=(8, 8), p=0.3),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2(),
        ])
    else:
        return A.Compose([
            A.Resize(height=image_size, width=image_size),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2(),
        ])


# ==================== METRICS ====================
def dice_score(preds, targets, smooth=1e-6):
    preds = torch.sigmoid(preds)
    preds = preds.view(-1)
    targets = targets.view(-1)
    intersection = (preds * targets).sum()
    return (2. * intersection + smooth) / (preds.sum() + targets.sum() + smooth)


# ==================== TRAINING ====================
def main():
    print("=" * 80)
    print("BOOST EXISTING ADENSEUNET TO 93%+ DICE SCORE")
    print("=" * 80)

    # Data
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

    print(f"Train set size: {len(train_dataset)}, Val set size: {len(val_dataset)}")

    # Model (load our existing best checkpoint!)
    model = ADenseUNet(in_channels=3, out_channels=1).to(DEVICE)
    # Use upgraded CombinedLoss from adenseunet_simple.py
    criterion = CombinedLoss(
        bce_weight=0.2, 
        dice_weight=0.6, 
        focal_weight=0.2
    )
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=50, T_mult=2)
    scaler = GradScaler()  # For mixed precision
    best_val_dice = 0.0

    # Load existing best checkpoint!
    try:
        checkpoint = torch.load(os.path.join(CHECKPOINT_DIR, "best_adenseunet_ultimate_fixed.pth"), map_location=DEVICE, weights_only=False)
        model.load_state_dict(checkpoint["model_state_dict"])
        print(f"Loaded existing checkpoint with Dice: {checkpoint['best_dice']:.4f}")
        best_val_dice = checkpoint["best_dice"]
    except Exception as e:
        print(f"Error loading checkpoint: {e}")
        try:
            print("Trying old checkpoint...")
            checkpoint = torch.load(os.path.join(CHECKPOINT_DIR, "best_adenseunet_ultimate.pth"), map_location=DEVICE, weights_only=False)
            model.load_state_dict(checkpoint["model_state_dict"], strict=False)
            print(f"Loaded old checkpoint with Dice: {checkpoint['best_dice']:.4f}")
            best_val_dice = checkpoint["best_dice"]
        except Exception as e2:
            print(f"Error loading old checkpoint: {e2}")
            print("Starting from scratch...")

    # Training loop
    print("\nStarting training...")
    for epoch in range(EPOCHS):
        model.train()
        train_loss = 0.0
        train_dice = 0.0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS} (Train)")
        for images, masks in pbar:
            images, masks = images.to(DEVICE), masks.to(DEVICE)

            optimizer.zero_grad()
            with autocast():
                outputs = model(images)
                loss = criterion(outputs, masks)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()

            # Metrics
            with torch.no_grad():
                dice = dice_score(outputs, masks)
                train_loss += loss.item() * images.size(0)
                train_dice += dice.item() * images.size(0)

            pbar.set_postfix({"Loss": f"{loss.item():.4f}", "Dice": f"{dice.item():.4f}"})

        scheduler.step()

        # Validation
        model.eval()
        val_loss = 0.0
        val_dice = 0.0

        with torch.no_grad():
            pbar = tqdm(val_loader, desc=f"Epoch {epoch+1}/{EPOCHS} (Val)")
            for images, masks in pbar:
                images, masks = images.to(DEVICE), masks.to(DEVICE)
                outputs = model(images)
                loss = criterion(outputs, masks)
                dice = dice_score(outputs, masks)
                val_loss += loss.item() * images.size(0)
                val_dice += dice.item() * images.size(0)
                pbar.set_postfix({"Val Loss": f"{loss.item():.4f}", "Val Dice": f"{dice.item():.4f}"})

        avg_train_loss = train_loss / len(train_loader.dataset)
        avg_train_dice = train_dice / len(train_loader.dataset)
        avg_val_loss = val_loss / len(val_loader.dataset)
        avg_val_dice = val_dice / len(val_loader.dataset)

        print(f"\nEpoch {epoch+1} Summary:")
        print(f"Train - Loss: {avg_train_loss:.4f}, Dice: {avg_train_dice:.4f}")
        print(f"Val   - Loss: {avg_val_loss:.4f}, Dice: {avg_val_dice:.4f}")

        if avg_val_dice > best_val_dice:
            best_val_dice = avg_val_dice
            torch.save({
                "epoch": epoch + 1,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "best_dice": best_val_dice
            }, os.path.join(CHECKPOINT_DIR, "best_adenseunet_boosted.pth"))
            print(f"\n[NEW BEST] Val Dice: {best_val_dice:.4f}")

            if best_val_dice >= TARGET_DICE:
                print("\n" + "=" * 80)
                print("CONGRATULATIONS! 93%+ DICE SCORE ACHIEVED!")
                print(f"Final Val Dice: {best_val_dice:.4f}")
                print("=" * 80)
                break

    print("\nTraining complete!")
    print(f"Best Val Dice Score: {best_val_dice:.4f}")


if __name__ == "__main__":
    main()

