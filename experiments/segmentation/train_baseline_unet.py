
import os
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset, Dataset
import albumentations as A
from albumentations.pytorch import ToTensorV2
import numpy as np
from PIL import Image
from tqdm import tqdm
import segmentation_models_pytorch as smp


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
IMAGE_SIZE = 320
BATCH_SIZE = 12
LEARNING_RATE = 1e-4
EPOCHS = 100
PATIENCE = 15
DATA_ROOT = "combined_dataset"
SPLIT_PATH = "checkpoints/ultimate_split_indices.json"
OUTPUT_DIR = "baseline_results"
os.makedirs(OUTPUT_DIR, exist_ok=True)


class PolypDataset(Dataset):
    def __init__(self, data_root, transform=None):
        self.transform = transform
        images_dir = os.path.join(data_root, "images")
        masks_dir = os.path.join(data_root, "masks")
        self.image_files = sorted([f for f in os.listdir(images_dir) if f.lower().endswith((".png", ".jpg", ".jpeg"))])
        self.images_dir = images_dir
        self.masks_dir = masks_dir

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_path = os.path.join(self.images_dir, self.image_files[idx])
        mask_path = os.path.join(self.masks_dir, self.image_files[idx])
        image = np.array(Image.open(img_path).convert("RGB"))
        mask = np.array(Image.open(mask_path).convert("L"), dtype=np.float32)
        mask = (mask > 127.5).astype(np.float32)
        if self.transform:
            aug = self.transform(image=image, mask=mask)
            image, mask = aug["image"], aug["mask"]
        return image, mask.unsqueeze(0)


def get_transforms(train=True):
    if train:
        return A.Compose([
            A.Resize(IMAGE_SIZE, IMAGE_SIZE),
            A.Affine(scale=(0.75, 1.25), translate_percent=(0.15, 0.15), rotate=(-45, 45), p=0.85),
            A.RandomRotate90(p=0.7),
            A.HorizontalFlip(p=0.7),
            A.VerticalFlip(p=0.4),
            A.RandomBrightnessContrast(brightness_limit=0.4, contrast_limit=0.4, p=0.8),
            A.GaussianBlur(blur_limit=(3, 7), p=0.4),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2(),
        ])
    else:
        return A.Compose([
            A.Resize(IMAGE_SIZE, IMAGE_SIZE),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2(),
        ])


def dice_score(outputs, masks):
    probs = torch.sigmoid(outputs)
    preds = (probs > 0.5).float()
    intersection = (preds * masks).sum()
    return (2. * intersection + 1e-6) / (preds.sum() + masks.sum() + 1e-6)


def compute_all_metrics(masks_np, probs_np):
    preds_np = (probs_np > 0.5).astype(np.float32)
    eps = 1e-8
    tp = np.sum((masks_np == 1) & (preds_np == 1))
    tn = np.sum((masks_np == 0) & (preds_np == 0))
    fp = np.sum((masks_np == 0) & (preds_np == 1))
    fn = np.sum((masks_np == 1) & (preds_np == 0))
    dice = (2 * tp) / (2 * tp + fp + fn + eps)
    iou = tp / (tp + fp + fn + eps)
    precision = tp / (tp + fp + eps)
    recall = tp / (tp + fn + eps)
    specificity = tn / (tn + fp + eps)
    accuracy = (tp + tn) / (tp + tn + fp + fn + eps)
    f1 = (2 * precision * recall) / (precision + recall + eps)
    return dice, iou, precision, recall, specificity, accuracy, f1


def main():
    print("=== Training plain UNet baseline ===")

    with open(SPLIT_PATH, "r") as f:
        split = json.load(f)

    full_ds = PolypDataset(DATA_ROOT)
    train_ds = Subset(PolypDataset(DATA_ROOT, get_transforms(train=True)), split["train"])
    val_ds = Subset(PolypDataset(DATA_ROOT, get_transforms(train=False)), split["val"])
    test_ds = Subset(PolypDataset(DATA_ROOT, get_transforms(train=False)), split["test"])

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    # plain UNet baseline (no pretrained encoder)
    model = smp.Unet(encoder_name="resnet34", encoder_weights=None, in_channels=3, classes=1).to(DEVICE)
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=20, T_mult=2, eta_min=1e-6)
    criterion = nn.BCEWithLogitsLoss()

    best_val_dice = 0.0
    patience_counter = 0
    train_loss_history = []
    val_dice_history = []

    for epoch in range(EPOCHS):
        model.train()
        epoch_loss = 0.0

        for imgs, masks in tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}"):
            imgs, masks = imgs.to(DEVICE), masks.to(DEVICE)
            optimizer.zero_grad()
            outputs = model(imgs)
            loss = criterion(outputs, masks)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * imgs.size(0)

        train_loss = epoch_loss / len(train_loader.dataset)
        train_loss_history.append(train_loss)

        model.eval()
        total_dice = 0.0
        with torch.no_grad():
            for imgs, masks in val_loader:
                imgs, masks = imgs.to(DEVICE), masks.to(DEVICE)
                outputs = model(imgs)
                total_dice += dice_score(outputs, masks).item() * imgs.size(0)
        val_dice = total_dice / len(val_loader.dataset)
        val_dice_history.append(val_dice)

        print(f"Epoch {epoch+1}: Train loss {train_loss:.4f}, Val Dice {val_dice:.4f}")

        if val_dice > best_val_dice:
            best_val_dice = val_dice
            patience_counter = 0
            torch.save(model.state_dict(), os.path.join(OUTPUT_DIR, "baseline_unet_best.pth"))
            torch.save({"train_loss": train_loss_history, "val_dice": val_dice_history}, os.path.join(OUTPUT_DIR, "baseline_training_history.pth"))
        else:
            patience_counter += 1
            if patience_counter >= PATIENCE:
                print("Early stopping!")
                break

        scheduler.step()

    # evaluate on test set
    model.load_state_dict(torch.load(os.path.join(OUTPUT_DIR, "baseline_unet_best.pth"), map_location=DEVICE))
    model.eval()

    all_metrics = []
    with torch.no_grad():
        for imgs, masks in test_loader:
            imgs, masks = imgs.to(DEVICE), masks.to(DEVICE)
            outputs = model(imgs)
            probs = torch.sigmoid(outputs).cpu().numpy()
            masks_np = masks.cpu().numpy()

            for i in range(probs.shape[0]):
                mask = masks_np[i, 0]
                prob = probs[i, 0]
                dice, iou, precision, recall, specificity, accuracy, f1 = compute_all_metrics(mask.flatten(), prob.flatten())
                all_metrics.append({
                    "dice": float(dice), "iou": float(iou),
                    "precision": float(precision), "recall": float(recall),
                    "specificity": float(specificity), "accuracy": float(accuracy), "f1": float(f1)
                })

    avg_metrics = {k: float(np.mean([m[k] for m in all_metrics])) for k in all_metrics[0]}
    results = {"avg_metrics": avg_metrics, "all_metrics": all_metrics}

    print("\n=== Test Set Metrics ===")
    for k, v in avg_metrics.items():
        print(f"{k}: {v:.4f}")

    with open(os.path.join(OUTPUT_DIR, "baseline_results.json"), "w") as f:
        json.dump(results, f, indent=2)
    print("\nSaved baseline results to baseline_results.json!")


if __name__ == "__main__":
    main()
