
"""
Ultimate Segmentation Training for 90%+ Dice Score
- Larger images, better augmentations, better loss, etc.
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
from adenseunet_simple import ADenseUNet
from polypgate.metrics import dice_score, iou_score


# Configuration
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
IMAGE_SIZE = 352  # Larger images for more detail
BATCH_SIZE = 8    # Smaller batch size for larger images
LEARNING_RATE = 1e-4
EPOCHS = 200
DATA_ROOT = "archive (1)/PNG"
CHECKPOINT_DIR = "checkpoints"
LOG_DIR = "logs"

os.makedirs(CHECKPOINT_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)


class TverskyLoss(nn.Module):
    def __init__(self, alpha=0.7, smooth=1e-6):
        super().__init__()
        self.alpha = alpha
        self.smooth = smooth

    def forward(self, probs, targets):
        probs = probs.view(-1)
        targets = targets.view(-1)
        true_pos = (probs * targets).sum()
        false_neg = ((1 - probs) * targets).sum()
        false_pos = (probs * (1 - targets)).sum()
        tversky = (true_pos + self.smooth) / (true_pos + self.alpha * false_neg + (1 - self.alpha) * false_pos + self.smooth)
        return 1 - tversky


class UltimateLoss(nn.Module):
    def __init__(self, bce_weight=0.3, tversky_weight=0.7):
        super().__init__()
        self.bce = nn.BCELoss()
        self.tversky = TverskyLoss(alpha=0.7)
        self.bce_weight = bce_weight
        self.tversky_weight = tversky_weight

    def forward(self, logits, targets):
        probs = torch.sigmoid(logits)
        bce_loss = self.bce(probs, targets)
        tversky_loss = self.tversky(probs, targets)
        return self.bce_weight * bce_loss + self.tversky_weight * tversky_loss


class PolypSegmentationDataset(Dataset):
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
        image = np.array(Image.open(img_path).convert("RGB"))
        mask_path = os.path.join(self.gt_dir, img_file)
        mask = np.array(Image.open(mask_path).convert("L"), dtype=np.float32)
        mask = mask / 255.0

        if self.transform:
            augmented = self.transform(image=image, mask=mask)
            image = augmented["image"]
            mask = augmented["mask"]

        return image, mask.unsqueeze(0)


def get_data_loaders(data_root, batch_size=8, image_size=352, val_split=0.2):
    # Heavy data augmentation for better generalization
    train_transform = A.Compose([
        A.Resize(height=image_size, width=image_size),
        A.ShiftScaleRotate(shift_limit=0.15, scale_limit=0.15, rotate_limit=45, p=0.5),
        A.RandomRotate90(p=0.5),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.3),
        A.RandomBrightnessContrast(brightness_limit=0.3, contrast_limit=0.3, p=0.5),
        A.GaussianBlur(blur_limit=(3, 7), p=0.3),
        A.GaussNoise(var_limit=(10.0, 50.0), p=0.2),
        A.ElasticTransform(p=0.25, alpha=150, sigma=150 * 0.05),
        A.GridDistortion(p=0.25),
        A.OpticalDistortion(p=0.2, distort_limit=0.05),
        A.HueSaturationValue(hue_shift_limit=10, sat_shift_limit=25, val_shift_limit=15, p=0.3),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2(),
    ])

    val_transform = A.Compose([
        A.Resize(height=image_size, width=image_size),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2(),
    ])

    full_dataset = PolypSegmentationDataset(data_root)
    dataset_size = len(full_dataset)
    val_size = int(val_split * dataset_size)
    train_size = dataset_size - val_size
    train_dataset, val_dataset = torch.utils.data.random_split(
        full_dataset, [train_size, val_size], generator=torch.Generator().manual_seed(42)
    )

    train_dataset.dataset.transform = train_transform
    val_dataset.dataset.transform = val_transform

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    return train_loader, val_loader


def train_epoch(model, loader, criterion, optimizer, scheduler, device):
    model.train()
    total_loss = 0.0
    total_dice = 0.0
    total_iou = 0.0

    pbar = tqdm(loader, desc="Training")
    for images, masks in pbar:
        images, masks = images.to(device), masks.to(device)

        optimizer.zero_grad()
        logits = model(images)
        probs = torch.sigmoid(logits)
        loss = criterion(logits, masks)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        dice = dice_score(probs, masks)
        iou = iou_score(probs, masks)

        total_loss += loss.item() * images.size(0)
        total_dice += dice.item() * images.size(0)
        total_iou += iou.item() * images.size(0)

        pbar.set_postfix({
            "Loss": f"{loss.item():.4f}",
            "Dice": f"{dice.item():.4f}"
        })

    scheduler.step()
    avg_loss = total_loss / len(loader.dataset)
    avg_dice = total_dice / len(loader.dataset)
    avg_iou = total_iou / len(loader.dataset)
    return avg_loss, avg_dice, avg_iou


def val_epoch(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    total_dice = 0.0
    total_iou = 0.0

    with torch.no_grad():
        pbar = tqdm(loader, desc="Validation")
        for images, masks in pbar:
            images, masks = images.to(device), masks.to(device)
            logits = model(images)
            probs = torch.sigmoid(logits)
            loss = criterion(logits, masks)

            dice = dice_score(probs, masks)
            iou = iou_score(probs, masks)

            total_loss += loss.item() * images.size(0)
            total_dice += dice.item() * images.size(0)
            total_iou += iou.item() * images.size(0)

            pbar.set_postfix({
                "Loss": f"{loss.item():.4f}",
                "Dice": f"{dice.item():.4f}"
            })

    avg_loss = total_loss / len(loader.dataset)
    avg_dice = total_dice / len(loader.dataset)
    avg_iou = total_iou / len(loader.dataset)
    return avg_loss, avg_dice, avg_iou


def main():
    print("=" * 60)
    print("ULTIMATE SEGMENTATION TRAINING - Target 90%+ Dice")
    print("=" * 60)

    print("\nLoading dataset...")
    train_loader, val_loader = get_data_loaders(
        DATA_ROOT, batch_size=BATCH_SIZE, image_size=IMAGE_SIZE
    )
    print(f"Train set size: {len(train_loader.dataset)}, Val set size: {len(val_loader.dataset)}")

    print("\nInitializing model...")
    model = ADenseUNet(in_channels=3, out_channels=1).to(DEVICE)

    # Load improved checkpoint if available
    try:
        checkpoint = torch.load(os.path.join(CHECKPOINT_DIR, "best_adenseunet_improved.pth"), map_location=DEVICE, weights_only=False)
        model.load_state_dict(checkpoint["model_state_dict"])
        print("Successfully loaded improved checkpoint as starting point!")
    except:
        print("No improved checkpoint found, starting from scratch")

    criterion = UltimateLoss(bce_weight=0.3, tversky_weight=0.7)
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=50, T_mult=2)

    print("\nStarting training...")
    # Start from our previous best
    best_dice = 0.8885

    for epoch in range(EPOCHS):
        print(f"\nEpoch {epoch+1}/{EPOCHS}")
        train_loss, train_dice, train_iou = train_epoch(
            model, train_loader, criterion, optimizer, scheduler, DEVICE
        )
        val_loss, val_dice, val_iou = val_epoch(
            model, val_loader, criterion, DEVICE
        )

        print(f"Train - Loss: {train_loss:.4f}, Dice: {train_dice:.4f}, IoU: {train_iou:.4f}")
        print(f"Val   - Loss: {val_loss:.4f}, Dice: {val_dice:.4f}, IoU: {val_iou:.4f}")

        if val_dice > best_dice:
            best_dice = val_dice
            torch.save({
                "epoch": epoch+1,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "best_dice": best_dice,
            }, os.path.join(CHECKPOINT_DIR, "best_adenseunet_ultimate.pth"))
            print(f"\n[NEW BEST] - Val Dice: {best_dice:.4f}!")
            print("Checkpoint saved: best_adenseunet_ultimate.pth")

    print("\n" + "=" * 60)
    print(f"TRAINING COMPLETE - Best Val Dice: {best_dice:.4f}")
    print("=" * 60)


if __name__ == "__main__":
    main()
