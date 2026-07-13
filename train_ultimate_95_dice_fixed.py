
"""
Fixed Ultimate Polyp Segmentation Training for 95%+ Dice Score.
Features:
- Improved ADenseUNet (with extra layers)
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

# Import our existing models
from adenseunet_simple import ADenseUNet

# Configuration
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
IMAGE_SIZE = 512
BATCH_SIZE = 4  # Smaller batch size for 512x512
LEARNING_RATE = 1e-4
EPOCHS = 300
DATA_ROOT = "archive (1)/PNG"
CHECKPOINT_DIR = "checkpoints"
LOG_DIR = "logs"

os.makedirs(CHECKPOINT_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)


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
    def __init__(self, alpha=0.25, gamma=2.0, smooth=1e-6):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.smooth = smooth

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
            A.ShiftScaleRotate(shift_limit=0.15, scale_limit=0.15, rotate_limit=45, p=0.7),
            A.RandomRotate90(p=0.7),
            A.HorizontalFlip(p=0.7),
            A.VerticalFlip(p=0.3),
            A.RandomBrightnessContrast(brightness_limit=0.3, contrast_limit=0.3, p=0.7),
            A.GaussianBlur(blur_limit=(3, 7), p=0.3),
            A.GaussNoise(var_limit=(10.0, 50.0), p=0.2),
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


# ==================== IMPROVED ADENSEUNET (MORE LAYERS) ====================
class ImprovedADenseUNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=1):
        super().__init__()
        
        # Encoder (bigger)
        self.init_conv = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )
        
        self.dense1 = DenseBlock(64, growth_rate=48, num_layers=6)  # 64+6*48=352
        self.trans1 = nn.Sequential(
            nn.BatchNorm2d(352),
            nn.ReLU(inplace=True),
            nn.Conv2d(352, 176, 1, bias=False),
            nn.MaxPool2d(2)
        )
        
        self.dense2 = DenseBlock(176, growth_rate=48, num_layers=8)  #176+8*48=560
        self.trans2 = nn.Sequential(
            nn.BatchNorm2d(560),
            nn.ReLU(inplace=True),
            nn.Conv2d(560, 280, 1, bias=False),
            nn.MaxPool2d(2)
        )
        
        self.dense3 = DenseBlock(280, growth_rate=48, num_layers=12)  #280+12*48=856
        self.trans3 = nn.Sequential(
            nn.BatchNorm2d(856),
            nn.ReLU(inplace=True),
            nn.Conv2d(856, 428, 1, bias=False),
            nn.MaxPool2d(2)
        )
        
        self.dense4 = DenseBlock(428, growth_rate=48, num_layers=16)  #428+16*48=1200
        self.trans4 = nn.Sequential(
            nn.BatchNorm2d(1200),
            nn.ReLU(inplace=True),
            nn.Conv2d(1200, 600, 1, bias=False),
            nn.MaxPool2d(2)
        )
        
        self.bottleneck = DoubleConv(600, 1200)
        
        # Decoder
        self.up4 = nn.ConvTranspose2d(1200, 600, kernel_size=2, stride=2)
        self.att4 = AttentionBlock(F_g=600, F_l=1200, F_int=300)
        self.dec4 = DoubleConv(1200 + 600, 600)
        
        self.up3 = nn.ConvTranspose2d(600, 428, kernel_size=2, stride=2)
        self.att3 = AttentionBlock(F_g=428, F_l=856, F_int=214)
        self.dec3 = DoubleConv(856 + 428, 428)
        
        self.up2 = nn.ConvTranspose2d(428, 280, kernel_size=2, stride=2)
        self.att2 = AttentionBlock(F_g=280, F_l=560, F_int=140)
        self.dec2 = DoubleConv(560 + 280, 280)
        
        self.up1 = nn.ConvTranspose2d(280, 176, kernel_size=2, stride=2)
        self.att1 = AttentionBlock(F_g=176, F_l=352, F_int=88)
        self.dec1 = DoubleConv(352 + 176, 176)
        
        self.final_up = nn.ConvTranspose2d(176, 64, kernel_size=4, stride=4)
        self.final_conv = nn.Conv2d(64, out_channels, kernel_size=1)
        
    def forward(self, x):
        # Encoder
        x1 = self.init_conv(x)
        x2 = self.dense1(x1)
        x2_t = self.trans1(x2)
        x3 = self.dense2(x2_t)
        x3_t = self.trans2(x3)
        x4 = self.dense3(x3_t)
        x4_t = self.trans3(x4)
        x5 = self.dense4(x4_t)
        x5_t = self.trans4(x5)
        
        # Bottleneck
        bn = self.bottleneck(x5_t)
        
        # Decoder
        d4 = self.up4(bn)
        x5_att = self.att4(d4, x5)
        d4 = torch.cat([d4, x5_att], dim=1)
        d4 = self.dec4(d4)
        
        d3 = self.up3(d4)
        x4_att = self.att3(d3, x4)
        d3 = torch.cat([d3, x4_att], dim=1)
        d3 = self.dec3(d3)
        
        d2 = self.up2(d3)
        x3_att = self.att2(d2, x3)
        d2 = torch.cat([d2, x3_att], dim=1)
        d2 = self.dec2(d2)
        
        d1 = self.up1(d2)
        x2_att = self.att1(d1, x2)
        d1 = torch.cat([d1, x2_att], dim=1)
        d1 = self.dec1(d1)
        
        d0 = self.final_up(d1)
        out = self.final_conv(d0)
        return out


# Re-define the helper classes so they're in this file
class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)


class AttentionBlock(nn.Module):
    def __init__(self, F_g, F_l, F_int):
        super().__init__()
        self.W_g = nn.Sequential(
            nn.Conv2d(F_g, F_int, kernel_size=1),
            nn.BatchNorm2d(F_int)
        )
        self.W_x = nn.Sequential(
            nn.Conv2d(F_l, F_int, kernel_size=1),
            nn.BatchNorm2d(F_int)
        )
        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, kernel_size=1),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, g, x):
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        psi = self.relu(g1 + x1)
        psi = self.psi(psi)
        return x * psi


class DenseBlock(nn.Module):
    def __init__(self, in_channels, growth_rate, num_layers, bn_size=4):
        super().__init__()
        self.layers = nn.ModuleList()
        for _ in range(num_layers):
            self.layers.append(nn.Sequential(
                nn.BatchNorm2d(in_channels),
                nn.ReLU(inplace=True),
                nn.Conv2d(in_channels, bn_size * growth_rate, kernel_size=1, bias=False),
                nn.BatchNorm2d(bn_size * growth_rate),
                nn.ReLU(inplace=True),
                nn.Conv2d(bn_size * growth_rate, growth_rate, kernel_size=3, padding=1, bias=False)
            ))
            in_channels += growth_rate
        self.out_channels = in_channels

    def forward(self, x):
        features = [x]
        for layer in self.layers:
            new_features = layer(torch.cat(features, 1))
            features.append(new_features)
        return torch.cat(features, 1)


# ==================== TRAINING ====================
def main():
    print("=" * 80)
    print("ULTIMATE POLYP SEGMENTATION TRAINING - TARGET 95%+ DICE")
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

    # Model, loss, optimizer, scheduler
    model = ImprovedADenseUNet(in_channels=3, out_channels=1).to(DEVICE)
    criterion = UltimateLoss()
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=50, T_mult=2)
    best_val_dice = 0.0

    # Try to load previous best checkpoint
    try:
        checkpoint = torch.load(os.path.join(CHECKPOINT_DIR, "best_improved_adenseunet.pth"), map_location=DEVICE, weights_only=False)
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        best_val_dice = checkpoint["best_dice"]
        print(f"Loaded previous best checkpoint with Dice: {best_val_dice:.4f}")
    except Exception as e:
        print("No previous checkpoint found, starting from scratch")

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

            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, masks)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            # Metrics
            with torch.no_grad():
                dice = dice_score(outputs, masks)
                train_loss += loss.item() * images.size(0)
                train_dice += dice.item() * images.size(0)

            pbar.set_postfix({"Loss": f"{loss.item():.4f}", "Dice": f"{dice.item():.4f}"})

        # Update scheduler
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

        # Calculate averages
        avg_train_loss = train_loss / len(train_loader.dataset)
        avg_train_dice = train_dice / len(train_loader.dataset)
        avg_val_loss = val_loss / len(val_loader.dataset)
        avg_val_dice = val_dice / len(val_loader.dataset)

        # Print epoch summary
        print(f"\nEpoch {epoch+1} Summary:")
        print(f"Train - Loss: {avg_train_loss:.4f}, Dice: {avg_train_dice:.4f}")
        print(f"Val   - Loss: {avg_val_loss:.4f}, Dice: {avg_val_dice:.4f}")

        # Save best model
        if avg_val_dice > best_val_dice:
            best_val_dice = avg_val_dice
            torch.save({
                "epoch": epoch + 1,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "best_dice": best_val_dice
            }, os.path.join(CHECKPOINT_DIR, "best_improved_adenseunet.pth"))
            print(f"\n[NEW BEST] Val Dice: {best_val_dice:.4f} - Checkpoint saved!")

            # Check if we reached our goal!
            if best_val_dice >= 0.95:
                print("\n" + "=" * 80)
                print("🎉 CONGRATULATIONS! WE REACHED 95%+ DICE SCORE! 🎉")
                print(f"Final Val Dice: {best_val_dice:.4f}")
                print("=" * 80)
                break

    print("\nTraining complete!")
    print(f"Best Val Dice Score: {best_val_dice:.4f}")


if __name__ == "__main__":
    main()

