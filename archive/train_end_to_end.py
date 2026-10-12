
"""
End-to-End Training for Polyp Detection
Trains the combined segmentation + classification model
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
from end_to_end_model import EndToEndPolypModel, FocalLoss
from polypgate.metrics import compute_classification_metrics, dice_score, iou_score


# Configuration
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
IMAGE_SIZE = 224
BATCH_SIZE = 16
LEARNING_RATE = 1e-4
EPOCHS = 50
DATA_ROOT = "hyper-kvasir-labeled-images (1)/labeled-images"
CHECKPOINT_DIR = "outputs/checkpoints"
LOG_DIR = "outputs/logs"

os.makedirs(CHECKPOINT_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)


class PolypClassificationDataset(Dataset):
    def __init__(self, data_root, transform=None):
        self.data_root = data_root
        self.transform = transform
        
        # Collect all images and labels
        self.samples = []
        
        # Lower GI tract - pathological findings (polyps = class 1)
        lower_polyps_dir = os.path.join(data_root, "lower-gi-tract", "pathological-findings", "polyps")
        if os.path.exists(lower_polyps_dir):
            for f in os.listdir(lower_polyps_dir):
                if f.lower().endswith(('.jpg', '.jpeg', '.png')):
                    self.samples.append((os.path.join(lower_polyps_dir, f), 1))
        
        # Lower GI tract - normal findings (class 0)
        lower_normal_dirs = [
            os.path.join(data_root, "lower-gi-tract", "anatomical-landmarks"),
            os.path.join(data_root, "lower-gi-tract", "normal-cecum"),
            os.path.join(data_root, "lower-gi-tract", "normal-z-line"),
        ]
        for normal_dir in lower_normal_dirs:
            if os.path.exists(normal_dir):
                for root, _, files in os.walk(normal_dir):
                    for f in files:
                        if f.lower().endswith(('.jpg', '.jpeg', '.png')):
                            self.samples.append((os.path.join(root, f), 0))
        
        print(f"Collected {len(self.samples)} samples")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        image = np.array(Image.open(img_path).convert("RGB"))
        
        if self.transform:
            augmented = self.transform(image=image)
            image = augmented["image"]
        
        return image, torch.tensor(label, dtype=torch.long)


def get_data_loaders(data_root, batch_size=16, image_size=224, val_split=0.15, test_split=0.15):
    # Train transforms
    train_transform = A.Compose([
        A.Resize(height=image_size, width=image_size),
        A.RandomRotate90(p=0.5),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.2),
        A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.3),
        A.GaussianBlur(blur_limit=(3, 7), p=0.2),
        A.GaussNoise(var_limit=(10.0, 50.0), p=0.1),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2(),
    ])
    
    # Val/test transforms
    val_transform = A.Compose([
        A.Resize(height=image_size, width=image_size),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2(),
    ])
    
    # Split dataset
    full_dataset = PolypClassificationDataset(data_root)
    dataset_size = len(full_dataset)
    test_size = int(test_split * dataset_size)
    val_size = int(val_split * dataset_size)
    train_size = dataset_size - val_size - test_size
    
    train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(
        full_dataset, [train_size, val_size, test_size],
        generator=torch.Generator().manual_seed(42)
    )
    
    # Apply transforms
    train_dataset.dataset.transform = train_transform
    val_dataset.dataset.transform = val_transform
    test_dataset.dataset.transform = val_transform
    
    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    
    print(f"Train: {len(train_dataset)}, Val: {len(val_dataset)}, Test: {len(test_dataset)}")
    return train_loader, val_loader, test_loader


def train_epoch(model, loader, criterion, optimizer, scaler, device):
    model.train()
    total_loss = 0.0
    all_preds = []
    all_labels = []
    all_probs = []
    
    pbar = tqdm(loader, desc="Training")
    for images, labels in pbar:
        images, labels = images.to(device), labels.to(device)
        
        optimizer.zero_grad()
        
        with autocast():
            logits = model(images)
            loss = criterion(logits, labels)
            probs = torch.softmax(logits, dim=1)
            preds = torch.argmax(probs, dim=1)
        
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        
        # Collect metrics
        total_loss += loss.item() * images.size(0)
        all_preds.extend(preds.detach().cpu().numpy())
        all_labels.extend(labels.detach().cpu().numpy())
        all_probs.extend(probs[:, 1].detach().cpu().numpy())
        
        pbar.set_postfix({"Loss": f"{loss.item():.4f}"})
    
    avg_loss = total_loss / len(loader.dataset)
    metrics = compute_classification_metrics(np.array(all_labels), np.array(all_preds), np.array(all_probs))
    return avg_loss, metrics


def val_epoch(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    all_preds = []
    all_labels = []
    all_probs = []
    
    with torch.no_grad():
        pbar = tqdm(loader, desc="Validation")
        for images, labels in pbar:
            images, labels = images.to(device), labels.to(device)
            
            with autocast():
                logits = model(images)
                loss = criterion(logits, labels)
                probs = torch.softmax(logits, dim=1)
                preds = torch.argmax(probs, dim=1)
            
            total_loss += loss.item() * images.size(0)
            all_preds.extend(preds.detach().cpu().numpy())
            all_labels.extend(labels.detach().cpu().numpy())
            all_probs.extend(probs[:, 1].detach().cpu().numpy())
            
            pbar.set_postfix({"Loss": f"{loss.item():.4f}"})
    
    avg_loss = total_loss / len(loader.dataset)
    metrics = compute_classification_metrics(np.array(all_labels), np.array(all_preds), np.array(all_probs))
    return avg_loss, metrics


def main():
    # Load data
    print("Loading dataset...")
    train_loader, val_loader, test_loader = get_data_loaders(
        DATA_ROOT, batch_size=BATCH_SIZE, image_size=IMAGE_SIZE
    )
    
    # Initialize model
    print("Initializing end-to-end model...")
    model = EndToEndPolypModel(num_classes=2, pretrained_vit=True, freeze_segmentation=True).to(DEVICE)
    
    # Loss, optimizer, scheduler
    criterion = FocalLoss(alpha=0.25, gamma=2.0)
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
    scaler = GradScaler()
    
    # Training loop
    print("Starting training...")
    best_auc = 0.0
    train_losses, val_losses = [], []
    train_aucs, val_aucs = [], []
    
    for epoch in range(EPOCHS):
        print(f"\nEpoch {epoch+1}/{EPOCHS}")
        
        # Train
        train_loss, train_metrics = train_epoch(
            model, train_loader, criterion, optimizer, scaler, DEVICE
        )
        
        # Val
        val_loss, val_metrics = val_epoch(
            model, val_loader, criterion, DEVICE
        )
        
        # Scheduler step
        scheduler.step()
        
        # Save metrics
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_aucs.append(train_metrics["roc_auc"])
        val_aucs.append(val_metrics["roc_auc"])
        
        # Print summary
        print(f"Train - Loss: {train_loss:.4f}, Acc: {train_metrics['accuracy']:.4f}, AUC: {train_metrics['roc_auc']:.4f}")
        print(f"Val   - Loss: {val_loss:.4f}, Acc: {val_metrics['accuracy']:.4f}, AUC: {val_metrics['roc_auc']:.4f}")
        
        # Save best model
        if val_metrics["roc_auc"] > best_auc:
            best_auc = val_metrics["roc_auc"]
            torch.save({
                "epoch": epoch+1,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "best_auc": best_auc,
                "metrics": val_metrics,
            }, os.path.join(CHECKPOINT_DIR, "best_end_to_end.pth"))
            print(f"--> Best model saved! Val AUC: {best_auc:.4f}")
    
    print("\nTraining complete!")
    print(f"Best Val AUC: {best_auc:.4f}")
    
    # Evaluate on test set
    print("\nEvaluating on test set...")
    checkpoint = torch.load(os.path.join(CHECKPOINT_DIR, "best_end_to_end.pth"), map_location=DEVICE, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    test_loss, test_metrics = val_epoch(model, test_loader, criterion, DEVICE)
    print("\nTest Results:")
    for k, v in test_metrics.items():
        if k != "confusion_matrix":
            print(f"  {k}: {v:.4f}")
    
    # Plot training curves
    import matplotlib.pyplot as plt
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    ax1.plot(train_losses, label="Train Loss")
    ax1.plot(val_losses, label="Val Loss")
    ax1.set_title("Loss Curve")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.legend()
    
    ax2.plot(train_aucs, label="Train AUC")
    ax2.plot(val_aucs, label="Val AUC")
    ax2.set_title("AUC Curve")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("ROC-AUC")
    ax2.legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(LOG_DIR, "end_to_end_training_curves.png"), dpi=300, bbox_inches="tight")
    print(f"Training curves saved to {LOG_DIR}")


if __name__ == "__main__":
    main()
