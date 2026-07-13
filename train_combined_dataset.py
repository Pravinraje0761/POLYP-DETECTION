
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
from adenseunet_boosted import ADenseUNet_Boosted

# Configuration
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
IMAGE_SIZE = 224
BATCH_SIZE = 8
EPOCHS = 200
LEARNING_RATE = 1e-4
DATA_ROOT = "combined_dataset"
CHECKPOINT_DIR = "checkpoints"
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

print(f"Using device: {DEVICE}")

# Dataset Class
class CombinedPolypDataset(Dataset):
    def __init__(self, data_root, transform=None):
        self.data_root = data_root
        self.images_dir = os.path.join(data_root, "images")
        self.masks_dir = os.path.join(data_root, "masks")
        self.transform = transform
        
        # Get all image files
        self.image_files = sorted([
            f for f in os.listdir(self.images_dir)
            if f.lower().endswith((".png", ".jpg", ".jpeg"))
        ])
        
        # Verify mask files exist
        for img_file in self.image_files:
            mask_path = os.path.join(self.masks_dir, img_file)
            if not os.path.exists(mask_path):
                raise FileNotFoundError(f"Mask not found for {img_file}")
        
        print(f"Loaded {len(self.image_files)} image-mask pairs")
    
    def __len__(self):
        return len(self.image_files)
    
    def __getitem__(self, idx):
        img_file = self.image_files[idx]
        
        # Load image
        img_path = os.path.join(self.images_dir, img_file)
        image = np.array(Image.open(img_path).convert("RGB"))
        
        # Load mask
        mask_path = os.path.join(self.masks_dir, img_file)
        mask = np.array(Image.open(mask_path).convert("L"), dtype=np.float32) / 255.0
        mask = (mask > 0.5).astype(np.float32)
        
        # Apply transforms
        if self.transform:
            augmented = self.transform(image=image, mask=mask)
            image = augmented["image"]
            mask = augmented["mask"]
        
        return image, mask.unsqueeze(0)

# Dice Score Function
def dice_score(logits, targets, smooth=1e-6):
    preds = torch.sigmoid(logits)
    preds = (preds > 0.5).float()
    preds = preds.view(-1)
    targets = targets.view(-1)
    intersection = (preds * targets).sum()
    return (2.0 * intersection + smooth) / (preds.sum() + targets.sum() + smooth)

# Combined Loss: Dice + BCE
class CombinedLoss(nn.Module):
    def __init__(self, smooth=1e-6):
        super(CombinedLoss, self).__init__()
        self.smooth = smooth
        self.bce = nn.BCEWithLogitsLoss()
    
    def forward(self, logits, targets):
        # Dice loss
        probs = torch.sigmoid(logits)
        probs_flat = probs.view(-1)
        targets_flat = targets.view(-1)
        intersection = (probs_flat * targets_flat).sum()
        dice_loss = 1 - (2.0 * intersection + self.smooth) / (probs_flat.sum() + targets_flat.sum() + self.smooth)
        
        # BCE loss
        bce_loss = self.bce(logits, targets)
        
        return 0.5 * bce_loss + 0.5 * dice_loss

# Data Transforms
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

# Split Dataset into Train/Val
full_dataset = CombinedPolypDataset(DATA_ROOT)
train_size = int(0.8 * len(full_dataset))
val_size = len(full_dataset) - train_size
train_dataset, val_dataset = torch.utils.data.random_split(
    full_dataset, [train_size, val_size], generator=torch.Generator().manual_seed(42)
)

# Apply transforms to datasets
train_dataset.dataset.transform = train_transform
val_dataset.dataset.transform = val_transform

# Create DataLoaders
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

print(f"Train set size: {len(train_dataset)}, Val set size: {len(val_dataset)}")

# Initialize Model
model = ADenseUNet_Boosted(in_channels=3, out_channels=1).to(DEVICE)

# Load best checkpoint if available
best_checkpoint_path = os.path.join(CHECKPOINT_DIR, "best_adenseunet_boosted.pth")
start_epoch = 0
best_val_dice = 0.0

if os.path.exists(best_checkpoint_path):
    print(f"Loading existing checkpoint from {best_checkpoint_path}")
    checkpoint = torch.load(best_checkpoint_path, map_location=DEVICE, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    if "best_dice" in checkpoint:
        best_val_dice = checkpoint["best_dice"]
    if "epoch" in checkpoint:
        start_epoch = checkpoint["epoch"]
    print(f"Loaded checkpoint with best dice: {best_val_dice:.6f}")

# Loss, Optimizer, Scheduler
criterion = CombinedLoss()
optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

# Training Loop
print("=" * 80)
print("Starting training on combined dataset")
print("=" * 80)

for epoch in range(start_epoch, EPOCHS):
    # Train phase
    model.train()
    train_loss = 0.0
    train_dice = 0.0
    
    train_pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS} (Train)")
    for images, masks in train_pbar:
        images, masks = images.to(DEVICE), masks.to(DEVICE)
        
        optimizer.zero_grad()
        logits = model(images)
        loss = criterion(logits, masks)
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
        with torch.no_grad():
            dice = dice_score(logits, masks)
            train_loss += loss.item() * images.size(0)
            train_dice += dice.item() * images.size(0)
        
        train_pbar.set_postfix({
            "Loss": f"{loss.item():.4f}",
            "Dice": f"{dice.item():.4f}"
        })
    
    avg_train_loss = train_loss / len(train_loader.dataset)
    avg_train_dice = train_dice / len(train_loader.dataset)
    
    # Validation phase
    model.eval()
    val_loss = 0.0
    val_dice = 0.0
    
    with torch.no_grad():
        val_pbar = tqdm(val_loader, desc=f"Epoch {epoch+1}/{EPOCHS} (Val)")
        for images, masks in val_pbar:
            images, masks = images.to(DEVICE), masks.to(DEVICE)
            
            logits = model(images)
            loss = criterion(logits, masks)
            dice = dice_score(logits, masks)
            
            val_loss += loss.item() * images.size(0)
            val_dice += dice.item() * images.size(0)
            
            val_pbar.set_postfix({
                "Val Loss": f"{loss.item():.4f}",
                "Val Dice": f"{dice.item():.4f}"
            })
    
    avg_val_loss = val_loss / len(val_loader.dataset)
    avg_val_dice = val_dice / len(val_loader.dataset)
    
    scheduler.step()
    
    print(f"\nEpoch {epoch+1}/{EPOCHS}:")
    print(f"  Train - Loss: {avg_train_loss:.4f}, Dice: {avg_train_dice:.4f}")
    print(f"  Val   - Loss: {avg_val_loss:.4f}, Dice: {avg_val_dice:.4f}")
    
    # Save best model
    if avg_val_dice > best_val_dice:
        best_val_dice = avg_val_dice
        torch.save({
            "epoch": epoch + 1,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "best_dice": best_val_dice
        }, os.path.join(CHECKPOINT_DIR, "best_adenseunet_combined.pth"))
        print(f"  NEW BEST MODEL SAVED! (Dice: {best_val_dice:.6f})")
    
    # Periodic save every 50 epochs
    if (epoch + 1) % 50 == 0:
        torch.save({
            "epoch": epoch + 1,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "best_dice": best_val_dice
        }, os.path.join(CHECKPOINT_DIR, f"checkpoint_combined_epoch_{epoch+1}.pth"))
        print(f"  Periodic checkpoint saved (epoch {epoch+1})")
    
    print()

print("=" * 80)
print("Training complete!")
print(f"Best validation Dice score: {best_val_dice:.6f}")
print("=" * 80)
