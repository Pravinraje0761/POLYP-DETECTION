
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
OUTPUT_DIR = "multi_seed_results"
os.makedirs(OUTPUT_DIR, exist_ok=True)
SEEDS = [42, 1337, 9999]


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


def main():
    with open(SPLIT_PATH, "r") as f:
        split = json.load(f)

    full_ds = PolypDataset(DATA_ROOT)
    val_ds = Subset(PolypDataset(DATA_ROOT, get_transforms(train=False)), split["val"])
    test_ds = Subset(PolypDataset(DATA_ROOT, get_transforms(train=False)), split["test"])
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    all_results = {"seeds": SEEDS, "test_dice": []}

    for seed in SEEDS:
        print(f"\n=== Training with seed {seed} ===")
        torch.manual_seed(seed)
        np.random.seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

        train_ds = Subset(PolypDataset(DATA_ROOT, get_transforms(train=True)), split["train"])
        train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)

        model = smp.UnetPlusPlus(encoder_name="efficientnet-b4", encoder_weights="imagenet", in_channels=3, classes=1).to(DEVICE)
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

            for imgs, masks in tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS} (Seed {seed})"):
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
                torch.save(model.state_dict(), os.path.join(OUTPUT_DIR, f"seed_{seed}_best.pth"))
            else:
                patience_counter += 1
                if patience_counter >= PATIENCE:
                    print("Early stopping!")
                    break

            scheduler.step()

        # evaluate on test set
        model.load_state_dict(torch.load(os.path.join(OUTPUT_DIR, f"seed_{seed}_best.pth"), map_location=DEVICE))
        model.eval()
        test_dice = 0.0
        with torch.no_grad():
            for imgs, masks in test_loader:
                imgs, masks = imgs.to(DEVICE), masks.to(DEVICE)
                outputs = model(imgs)
                test_dice += dice_score(outputs, masks).item() * imgs.size(0)
        test_dice /= len(test_loader.dataset)

        all_results["test_dice"].append(test_dice)
        print(f"Seed {seed}: Test Dice {test_dice:.4f}")

    # compute mean and std
    all_results["mean_test_dice"] = float(np.mean(all_results["test_dice"]))
    all_results["std_test_dice"] = float(np.std(all_results["test_dice"]))
    print(f"\nMean test Dice: {all_results['mean_test_dice']:.4f}, Std: {all_results['std_test_dice']:.4f}")

    with open(os.path.join(OUTPUT_DIR, "multi_seed_results.json"), "w") as f:
        json.dump(all_results, f, indent=2)
    print("Saved results to multi_seed_results.json!")


if __name__ == "__main__":
    main()
