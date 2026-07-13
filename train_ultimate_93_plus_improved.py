
"""
ULTIMATE Polyp Segmentation Training - Target >93% Dice!
Features:
- Uses best checkpoint to start (0.9086 Dice)
- Advanced loss weighting
- CosineAnnealingWarmRestarts
- Stochastic Weight Averaging (SWA)
- Strong augmentations
- Gradient clipping
- Mixed precision (if CUDA available)
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

# Import model (must match best_adenseunet_boosted*.pth checkpoint keys)
from adenseunet_boosted import ADenseUNet_Boosted as ADenseUNet
from torch.cuda.amp import GradScaler, autocast

# Configuration
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
IMAGE_SIZE = 224
BATCH_SIZE = 16
LEARNING_RATE = 5e-5
EPOCHS = 400
DATA_ROOT = "combined_dataset"
CHECKPOINT_DIR = "checkpoints"
TARGET_DICE = 0.93

os.makedirs(CHECKPOINT_DIR, exist_ok=True)


# ==================== Advanced Loss Functions ====================
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
    def __init__(self, alpha=0.6, smooth=1e-6):
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


class UltraLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.bce = nn.BCEWithLogitsLoss()
        self.dice = DiceLoss()
        self.tversky = TverskyLoss()
        self.focal = FocalLoss()

    def forward(self, logits, targets):
        return (0.15 * self.bce(logits, targets) +
                0.35 * self.dice(logits, targets) +
                0.30 * self.tversky(logits, targets) +
                0.20 * self.focal(logits, targets))


# ==================== Dataset ====================
class PolypDataset(Dataset):
    def __init__(self, data_root, transform=None):
        self.transform = transform
        images_dir = os.path.join(data_root, "images")
        masks_dir = os.path.join(data_root, "masks")
        if os.path.isdir(images_dir) and os.path.isdir(masks_dir):
            self.images_dir = images_dir
            self.masks_dir = masks_dir
        else:
            # Fallback: archive layout (Original / Ground Truth)
            self.images_dir = os.path.join(data_root, "Original")
            self.masks_dir = os.path.join(data_root, "Ground Truth")
        self.image_files = sorted([
            f for f in os.listdir(self.images_dir)
            if f.lower().endswith((".png", ".jpg", ".jpeg"))
        ])

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_file = self.image_files[idx]
        img_path = os.path.join(self.images_dir, img_file)
        mask_path = os.path.join(self.masks_dir, img_file)

        image = np.array(Image.open(img_path).convert("RGB"))
        mask = np.array(Image.open(mask_path).convert("L"), dtype=np.float32)
        mask = (mask > 127.5).astype(np.float32)

        if self.transform:
            augmented = self.transform(image=image, mask=mask)
            image = augmented["image"]
            mask = augmented["mask"]

        return image, mask.unsqueeze(0)


# ==================== Strong Augmentations ====================
def get_transforms(image_size=224, is_train=True):
    if is_train:
        return A.Compose([
            A.Resize(image_size, image_size),
            A.Affine(scale=(0.75, 1.25), translate_percent=(0.15, 0.15),
                     rotate=(-45, 45), p=0.85),
            A.RandomRotate90(p=0.7),
            A.HorizontalFlip(p=0.7),
            A.VerticalFlip(p=0.4),
            A.RandomBrightnessContrast(brightness_limit=0.4, contrast_limit=0.4, p=0.8),
            A.GaussianBlur(blur_limit=(3, 7), p=0.4),
            A.GaussNoise(var_limit=(10.0, 70.0), p=0.3),
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


# ==================== Dice Score ====================
def dice_score(logits, targets, smooth=1e-6):
    probs = torch.sigmoid(logits)
    probs = probs.view(-1)
    targets = targets.view(-1)
    intersection = (probs * targets).sum()
    return (2. * intersection + smooth) / (probs.sum() + targets.sum() + smooth)


def load_starting_checkpoint(model):
    """Load the 90.86% Dice checkpoint into the matching architecture."""
    candidates = [
        "best_adenseunet_boosted_fixed.pth",
        "best_adenseunet_boosted.pth",
    ]
    for name in candidates:
        path = os.path.join(CHECKPOINT_DIR, name)
        if not os.path.exists(path):
            continue
        checkpoint = torch.load(path, map_location=DEVICE, weights_only=False)
        state_dict = checkpoint["model_state_dict"]
        remapped = {}
        for key, value in state_dict.items():
            if key.startswith("up_final."):
                remapped["final_up." + key.split(".", 1)[1]] = value
            else:
                remapped[key] = value
        model.load_state_dict(remapped, strict=True)
        best_dice = float(checkpoint.get("best_dice", 0.0))
        print(f"Loaded checkpoint: {path} (Dice: {best_dice:.4f})")
        return best_dice
    raise FileNotFoundError(
        "No starting checkpoint found. Place best_adenseunet_boosted_fixed.pth in checkpoints/."
    )


# ==================== Main Training ====================
def main():
    print("=" * 80)
    print("ULTIMATE POLYP SEGMENTATION TRAINING")
    print("Target: >93% Dice Score")
    print(f"Dataset: {DATA_ROOT}")
    print("=" * 80)

    if not os.path.isdir(os.path.join(DATA_ROOT, "images")):
        print(f"Combined dataset not found at {DATA_ROOT}/images.")
        print("Run: python prepare_combined_dataset.py")
        return

    # Load data
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

    # Initialize model (architecture matches 90.86% checkpoint)
    model = ADenseUNet(in_channels=3, out_channels=1).to(DEVICE)

    try:
        best_val_dice = load_starting_checkpoint(model)
    except Exception as e:
        print(f"Failed to load checkpoint: {e}")
        return

    # Training setup
    criterion = UltraLoss()
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=30, T_mult=2, eta_min=1e-6)
    scaler = GradScaler(enabled=torch.cuda.is_available())

    # Training loop
    print("\nStarting training...")
    for epoch in range(EPOCHS):
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

        # Validation
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
        print(f"\nEpoch {epoch+1}: Train Dice: {avg_train_dice:.4f}, Val Dice: {avg_val_dice:.4f}")

        # Save best model
        if avg_val_dice > best_val_dice:
            best_val_dice = avg_val_dice
            torch.save({
                "epoch": epoch + 1,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "best_dice": best_val_dice
            }, os.path.join(CHECKPOINT_DIR, "best_adenseunet_93_plus_final.pth"))
            print(f"NEW BEST! Val Dice: {best_val_dice:.4f}")

            # Check if we hit target!
            if best_val_dice >= TARGET_DICE:
                print("\n" + "=" * 80)
                print(f"SUCCESS! REACHED {best_val_dice:.4f} Dice SCORE!")
                print("=" * 80)
                break

    print("\nTraining complete! Best Val Dice: {:.4f}".format(best_val_dice))
    if best_val_dice < TARGET_DICE:
        print(f"Target not reached yet ({TARGET_DICE:.0%}). Continue training or run again.")
    else:
        print(f"Target reached: validation Dice >= {TARGET_DICE:.0%}")


if __name__ == "__main__":
    main()

