
"""
Stage 1 Training: A-DenseUNet for Polyp Segmentation
Uses the dataset from 'archive (1)' directory
"""
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.cuda.amp import GradScaler, autocast
import albumentations as A
from albumentations.pytorch import ToTensorV2
import numpy as np
from PIL import Image
from tqdm import tqdm
from adenseunet_simple import ADenseUNet, CombinedLoss
from polypgate.metrics import dice_score, iou_score

# Configuration
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
IMAGE_SIZE = 224
BATCH_SIZE = 16
LEARNING_RATE = 1e-4
EPOCHS = 100
DATA_ROOT = "archive (1)/PNG"
CHECKPOINT_DIR = "checkpoints"
LOG_DIR = "logs"

os.makedirs(CHECKPOINT_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)


class PolypSegmentationDataset(Dataset):
    def __init__(self, data_root, transform=None):
        self.data_root = data_root
        self.original_dir = os.path.join(data_root, "Original")
        self.gt_dir = os.path.join(data_root, "Ground Truth")
        self.transform = transform
        # Get all image files
        self.image_files = sorted([f for f in os.listdir(self.original_dir) if f.endswith(".png")])

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_file = self.image_files[idx]
        # Load image
        img_path = os.path.join(self.original_dir, img_file)
        image = np.array(Image.open(img_path).convert("RGB"))
        # Load mask
        mask_path = os.path.join(self.gt_dir, img_file)
        mask = np.array(Image.open(mask_path).convert("L"), dtype=np.float32)
        # Normalize mask to [0,1]
        mask = mask / 255.0

        if self.transform:
            augmented = self.transform(image=image, mask=mask)
            image = augmented["image"]
            mask = augmented["mask"]

        return image, mask.unsqueeze(0)


def get_data_loaders(data_root, batch_size=16, image_size=224, val_split=0.2):
    # Train transforms
    train_transform = A.Compose([
        A.Resize(height=image_size, width=image_size),
        A.RandomRotate90(p=0.5),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.2),
        A.RandomBrightnessContrast(p=0.3),
        A.GaussianBlur(blur_limit=(3, 7), p=0.2),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2(),
    ])

    # Val transforms
    val_transform = A.Compose([
        A.Resize(height=image_size, width=image_size),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2(),
    ])

    # Split dataset
    full_dataset = PolypSegmentationDataset(data_root)
    dataset_size = len(full_dataset)
    val_size = int(val_split * dataset_size)
    train_size = dataset_size - val_size
    train_dataset, val_dataset = torch.utils.data.random_split(
        full_dataset, [train_size, val_size], generator=torch.Generator().manual_seed(42)
    )

    # Apply transforms
    train_dataset.dataset.transform = train_transform
    val_dataset.dataset.transform = val_transform

    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    return train_loader, val_loader


def train_epoch(model, loader, criterion, optimizer, scaler, device):
    model.train()
    total_loss = 0.0
    total_dice = 0.0
    total_iou = 0.0

    pbar = tqdm(loader, desc="Training")
    for images, masks in pbar:
        images, masks = images.to(device), masks.to(device)

        optimizer.zero_grad()

        with autocast():
            outputs = model(images)
            probs = torch.sigmoid(outputs)
            loss = criterion(outputs, masks)  # criterion applies sigmoid internally

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        # Metrics
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


def val_epoch(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    total_dice = 0.0
    total_iou = 0.0

    with torch.no_grad():
        pbar = tqdm(loader, desc="Validation")
        for images, masks in pbar:
            images, masks = images.to(device), masks.to(device)

            with autocast():
                outputs = model(images)
                probs = torch.sigmoid(outputs)
                loss = criterion(outputs, masks)

            # Metrics
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
    # Load data
    print("Loading dataset...")
    train_loader, val_loader = get_data_loaders(
        DATA_ROOT, batch_size=BATCH_SIZE, image_size=IMAGE_SIZE
    )
    print(f"Train set size: {len(train_loader.dataset)}, Val set size: {len(val_loader.dataset)}")

    # Initialize model
    print("Initializing A-DenseUNet...")
    model = ADenseUNet(in_channels=3, out_channels=1).to(DEVICE)

    # Loss, optimizer, scheduler
    criterion = CombinedLoss(bce_weight=0.5, dice_weight=0.5)
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
    scaler = GradScaler()

    # Training loop
    print("Starting training...")
    best_dice = 0.0
    train_losses, val_losses = [], []
    train_dices, val_dices = [], []

    for epoch in range(EPOCHS):
        print(f"\nEpoch {epoch+1}/{EPOCHS}")
        # Train
        train_loss, train_dice, train_iou = train_epoch(
            model, train_loader, criterion, optimizer, scaler, DEVICE
        )
        # Val
        val_loss, val_dice, val_iou = val_epoch(
            model, val_loader, criterion, DEVICE
        )

        # Scheduler step
        scheduler.step()

        # Save metrics
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_dices.append(train_dice)
        val_dices.append(val_dice)

        # Print summary
        print(f"Train - Loss: {train_loss:.4f}, Dice: {train_dice:.4f}, IoU: {train_iou:.4f}")
        print(f"Val   - Loss: {val_loss:.4f}, Dice: {val_dice:.4f}, IoU: {val_iou:.4f}")

        # Save best model
        if val_dice > best_dice:
            best_dice = val_dice
            torch.save({
                "epoch": epoch+1,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "best_dice": best_dice,
            }, os.path.join(CHECKPOINT_DIR, "best_adenseunet.pth"))
            print(f"--> Best model saved! Val Dice: {best_dice:.4f}")

    print("\nTraining complete!")
    print(f"Best Val Dice Score: {best_dice:.4f}")

    # Plot training curves
    import matplotlib.pyplot as plt
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

    ax1.plot(train_losses, label="Train Loss")
    ax1.plot(val_losses, label="Val Loss")
    ax1.set_title("Loss Curve")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.legend()

    ax2.plot(train_dices, label="Train Dice")
    ax2.plot(val_dices, label="Val Dice")
    ax2.set_title("Dice Score Curve")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Dice Score")
    ax2.legend()

    plt.tight_layout()
    plt.savefig(os.path.join(LOG_DIR, "stage1_training_curves.png"), dpi=300, bbox_inches="tight")
    plt.savefig(os.path.join(LOG_DIR, "stage1_training_curves.pdf"))
    print(f"Training curves saved to {LOG_DIR}")


if __name__ == "__main__":
    main()
