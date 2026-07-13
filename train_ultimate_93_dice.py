
"""
Polyp Segmentation Training for 93%+ Dice Score!
Features:
  - Larger image size: 512x512
  - Heavy data augmentations (colonoscopy-specific)
  - Advanced loss function: BCE + Dice + Focal + Tversky
  - Deep supervision
  - Cyclic LR scheduler
  - Mixed precision training
  - Starts from existing best checkpoint!
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
from torch.cuda.amp import autocast, GradScaler

# Import our original model!
from adenseunet_simple import ADenseUNet

# Configuration
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
IMAGE_SIZE = 512
BATCH_SIZE = 4
LEARNING_RATE = 1e-4
EPOCHS = 300
DATA_ROOT = "archive (1)/PNG"
CHECKPOINT_DIR = "checkpoints"
BEST_CHECKPOINT = os.path.join(CHECKPOINT_DIR, "best_adenseunet_ultimate.pth")
NEW_BEST_CHECKPOINT = os.path.join(CHECKPOINT_DIR, "best_adenseunet_93_dice.pth")
LOG_DIR = "logs"

os.makedirs(CHECKPOINT_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)


# ------------------------------
# LOSS FUNCTIONS (ENHANCED)
# ------------------------------
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
        bce_w=0.1,
        dice_w=0.4,
        focal_w=0.25,
        tversky_w=0.25
    ):
        super().__init__()
        self.bce = nn.BCEWithLogitsLoss()
        self.dice = DiceLoss()
        self.focal = FocalLoss()
        self.tversky = TverskyLoss()
        self.bce_w = bce_w
        self.dice_w = dice_w
        self.focal_w = focal_w
        self.tversky_w = tversky_w

    def forward(self, logits, targets):
        return (
            self.bce_w * self.bce(logits, targets) +
            self.dice_w * self.dice(logits, targets) +
            self.focal_w * self.focal(logits, targets) +
            self.tversky_w * self.tversky(logits, targets)
        )


# ------------------------------
# DATASET
# ------------------------------
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
        mask = (mask > 127.5).astype(np.float32)

        if self.transform:
            augmented = self.transform(image=image, mask=mask)
            image = augmented["image"]
            mask = augmented["mask"]

        return image, mask.unsqueeze(0)


# ------------------------------
# DATA AUGMENTATIONS (SUPER POWERED!)
# ------------------------------
def get_transforms(image_size=512, is_train=True):
    if is_train:
        return A.Compose([
            A.Resize(height=image_size, width=image_size),
            A.Affine(scale=(0.75, 1.25), translate_percent=(0.2, 0.2), rotate=(-45, 45), p=0.75),
            A.RandomRotate90(p=0.7),
            A.HorizontalFlip(p=0.7),
            A.VerticalFlip(p=0.4),
            A.RandomBrightnessContrast(brightness_limit=0.35, contrast_limit=0.35, p=0.8),
            A.CLAHE(clip_limit=3.0, tile_grid_size=(8, 8), p=0.5),
            A.GaussianBlur(blur_limit=(3, 7), p=0.4),
            A.ElasticTransform(p=0.4, alpha=200, sigma=200 * 0.05),
            A.GridDistortion(p=0.3),
            A.OpticalDistortion(p=0.3, distort_limit=0.06),
            A.HueSaturationValue(hue_shift_limit=20, sat_shift_limit=40, val_shift_limit=30, p=0.5),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2(),
        ])
    else:
        return A.Compose([
            A.Resize(height=image_size, width=image_size),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2(),
        ])


# ------------------------------
# METRICS
# ------------------------------
def dice_score(preds, targets, smooth=1e-6):
    preds = torch.sigmoid(preds)
    preds = preds.view(-1)
    targets = targets.view(-1)
    intersection = (preds * targets).sum()
    return (2. * intersection + smooth) / (preds.sum() + targets.sum() + smooth)


# ------------------------------
# TRAINING LOOP
# ------------------------------
def main():
    print("=" * 80)
    print("TRAINING FOR 93%+ DICE SCORE!")
    print("=" * 80)

    # Data preparation
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
    print(f"Train size: {len(train_dataset)}, Val size: {len(val_dataset)}")

    # Model
    model = ADenseUNet(in_channels=3, out_channels=1).to(DEVICE)

    # Load existing best checkpoint!
    best_val_dice = 0.0
    start_epoch = 0
    if os.path.exists(BEST_CHECKPOINT):
        print(f"\nLoading existing checkpoint: {BEST_CHECKPOINT}")
        checkpoint = torch.load(BEST_CHECKPOINT, map_location=DEVICE, weights_only=False)
        model.load_state_dict(checkpoint["model_state_dict"])
        if "best_dice" in checkpoint:
            best_val_dice = checkpoint["best_dice"]
            print(f"Starting best Dice: {best_val_dice:.4f}")
        if "epoch" in checkpoint:
            start_epoch = checkpoint["epoch"]
            print(f"Starting epoch: {start_epoch + 1}")
    else:
        print("No existing checkpoint found, starting from scratch!")

    # Loss, optimizer, scheduler, scaler (for mixed precision)
    criterion = UltimateLoss()
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.CyclicLR(
        optimizer,
        base_lr=1e-5,
        max_lr=1e-4,
        step_size_up=10,
        mode='triangular2',
        cycle_momentum=False
    )
    scaler = GradScaler()

    # Training
    for epoch in range(start_epoch, EPOCHS):
        print(f"\n--- Epoch {epoch + 1}/{EPOCHS} ---")

        # Train
        model.train()
        train_loss = 0.0
        train_dice = 0.0
        train_pbar = tqdm(train_loader, desc="Training")
        for images, masks in train_pbar:
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
            train_pbar.set_postfix(loss=f"{loss.item():.4f}", dice=f"{dice.item():.4f}")

        scheduler.step()
        avg_train_loss = train_loss / len(train_loader.dataset)
        avg_train_dice = train_dice / len(train_loader.dataset)

        # Validate
        model.eval()
        val_loss = 0.0
        val_dice = 0.0
        with torch.no_grad():
            val_pbar = tqdm(val_loader, desc="Validation")
            for images, masks in val_pbar:
                images, masks = images.to(DEVICE), masks.to(DEVICE)
                outputs = model(images)
                loss = criterion(outputs, masks)
                dice = dice_score(outputs, masks)
                val_loss += loss.item() * images.size(0)
                val_dice += dice.item() * images.size(0)
                val_pbar.set_postfix(loss=f"{loss.item():.4f}", dice=f"{dice.item():.4f}")

        avg_val_loss = val_loss / len(val_loader.dataset)
        avg_val_dice = val_dice / len(val_loader.dataset)

        print(f"\nTrain Loss: {avg_train_loss:.4f}, Train Dice: {avg_train_dice:.4f}")
        print(f"Val Loss: {avg_val_loss:.4f}, Val Dice: {avg_val_dice:.4f}")

        # Save best model!
        if avg_val_dice > best_val_dice:
            best_val_dice = avg_val_dice
            torch.save({
                "epoch": epoch + 1,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "best_dice": best_val_dice
            }, NEW_BEST_CHECKPOINT)
            print(f"\nNEW BEST DICE: {best_val_dice:.4f} (saved to {NEW_BEST_CHECKPOINT})!")

            # Check if we hit our target!
            if best_val_dice >= 0.93:
                print("\n" + "=" * 80)
                print("SUCCESS! DICE SCORE EXCEEDED 93%!")
                print(f"Final Best Dice: {best_val_dice:.4f}")
                print("=" * 80)
                break

    # Final summary
    print("\n" + "=" * 80)
    print("TRAINING COMPLETE!")
    print(f"Final Best Dice Score: {best_val_dice:.4f}")
    print(f"Checkpoint saved at: {NEW_BEST_CHECKPOINT}")
    print("=" * 80)


if __name__ == "__main__":
    import torch.nn.functional as F
    main()
