
import os
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split, WeightedRandomSampler
import numpy as np
import albumentations as A
from albumentations.pytorch import ToTensorV2
from PIL import Image
from tqdm import tqdm
from collections import defaultdict

# Import our files
from adenseunet_boosted import ADenseUNet_Boosted
from metrics import compute_segmentation_metrics
from loss_functions import CombinedLoss

# ================================ CONFIG ================================
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
IMAGE_SIZE = 224
BATCH_SIZE = 8
NUM_EPOCHS = 200  # 150-250 range
INITIAL_LR = 1e-4
FINAL_LR = 5e-5
DATA_ROOT = "combined_dataset"
CHECKPOINT_DIR = "checkpoints"
LOG_DIR = "logs_improved"
os.makedirs(CHECKPOINT_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)

# Loss parameters
FT_ALPHA = 0.3
FT_BETA = 0.7
FT_GAMMA = 0.75
LAMBDA_FT = 0.7
LAMBDA_BOUNDARY = 0.3

# Thresholds to evaluate
EVAL_THRESHOLDS = [0.35, 0.40, 0.45, 0.50]

# Dataset name mapping for our combined dataset
DATASET_PREFIXES = {
    "archive": "archive",
    "cvc_clinicdb": "cvc_colondb",  # Your file prefix is cvc_colondb
    "kvasir": "kvasir",
    "etis_larib": "etis",  # If present
    "cvc_colondb": "cvc_colondb",
}

print("="*120)
print("POLYP SEGMENTATION - IMPROVED TRAINING PIPELINE")
print("="*120)
print(f"Device: {DEVICE}")
print(f"Epochs: {NUM_EPOCHS}")
print(f"Initial LR: {INITIAL_LR}, Final LR: {FINAL_LR}")
print(f"Batch Size: {BATCH_SIZE}")
print("="*120)


# ================================ DATASET ================================
class CombinedPolypDataset(Dataset):
    def __init__(self, data_root, transform=None):
        self.images_dir = os.path.join(data_root, "images")
        self.masks_dir = os.path.join(data_root, "masks")
        self.transform = transform

        # Collect all files and group by dataset
        self.file_list = sorted([
            f for f in os.listdir(self.images_dir)
            if f.lower().endswith((".png", ".jpg", ".jpeg"))
        ])
        self.dataset_groups = defaultdict(list)
        for f in self.file_list:
            if f.startswith("archive"):
                self.dataset_groups["archive"].append(f)
            elif f.startswith("cvc_colondb"):
                self.dataset_groups["cvc_clinicdb"].append(f)
                self.dataset_groups["cvc_colondb"].append(f)  # Both names
            elif f.startswith("kvasir"):
                self.dataset_groups["kvasir"].append(f)
            elif f.startswith("etis"):
                self.dataset_groups["etis_larib"].append(f)
            else:
                self.dataset_groups["other"].append(f)

        print(f"Loaded {len(self.file_list)} images:")
        for ds, files in self.dataset_groups.items():
            print(f"  - {ds}: {len(files)}")

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        filename = self.file_list[idx]
        image_path = os.path.join(self.images_dir, filename)
        mask_path = os.path.join(self.masks_dir, filename)

        image = np.array(Image.open(image_path).convert("RGB"))
        mask = np.array(Image.open(mask_path).convert("L"), dtype=np.float32) / 255.0
        mask = (mask > 0.5).astype(np.float32)

        if self.transform:
            augmented = self.transform(image=image, mask=mask)
            image = augmented["image"]
            mask = augmented["mask"]

        # Determine which dataset this is from
        dataset_name = "other"
        if filename.startswith("archive"):
            dataset_name = "archive"
        elif filename.startswith("cvc_colondb"):
            dataset_name = "cvc_clinicdb"
        elif filename.startswith("kvasir"):
            dataset_name = "kvasir"
        elif filename.startswith("etis"):
            dataset_name = "etis_larib"

        return image, mask.unsqueeze(0), dataset_name


def get_data_loaders():
    # --- Train transforms (strong augmentation) ---
    train_transform = A.Compose([
        A.Resize(IMAGE_SIZE, IMAGE_SIZE),
        A.RandomRotate90(p=0.8),
        A.HorizontalFlip(p=0.7),
        A.VerticalFlip(p=0.5),
        A.Affine(
            scale=(0.8, 1.2),
            translate_percent=(0.1, 0.1),
            rotate=(-30, 30),
            shear=(-10, 10),
            p=0.7
        ),
        A.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1, p=0.7),
        A.HueSaturationValue(
            hue_shift_limit=20,
            sat_shift_limit=30,
            val_shift_limit=20,
            p=0.5
        ),
        A.RandomGamma(gamma_limit=(70, 130), p=0.5),
        A.RGBShift(r_shift_limit=20, g_shift_limit=20, b_shift_limit=20, p=0.5),
        A.GaussianBlur(blur_limit=(3, 5), p=0.3),
        A.CLAHE(clip_limit=2.0, tile_grid_size=(8, 8), p=0.3),
        A.GaussNoise(var_limit=(10, 50), p=0.3),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2(),
    ])

    # --- Val transforms (only resize and normalize) ---
    val_transform = A.Compose([
        A.Resize(IMAGE_SIZE, IMAGE_SIZE),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2(),
    ])

    # --- Load full dataset and split ---
    full_ds = CombinedPolypDataset(DATA_ROOT)
    train_size = int(0.8 * len(full_ds))
    val_size = len(full_ds) - train_size
    train_ds_full, val_ds_full = random_split(
        full_ds, [train_size, val_size],
        generator=torch.Generator().manual_seed(42)
    )

    # --- Apply transforms ---
    train_ds_full.dataset.transform = train_transform
    val_ds_full.dataset.transform = val_transform

    # --- Create balanced sampler for training ---
    # Assign weights based on dataset (over-sample smaller/difficult datasets)
    # Recommended weights (higher = more samples):
    # ETIS-Larib: 2.5x, CVC-ClinicDB/CVC-ColonDB: 1.5x, Kvasir/Archive: 1x
    train_file_indices = train_ds_full.indices
    weights = []
    for idx in train_file_indices:
        filename = full_ds.file_list[idx]
        if filename.startswith("etis"):
            weights.append(2.5)  # Hardest, highest weight
        elif filename.startswith("cvc_colondb"):
            weights.append(1.5)  # Medium
        else:  # archive, kvasir
            weights.append(1.0)

    weights = torch.tensor(weights, dtype=torch.float32)
    sampler = WeightedRandomSampler(
        weights=weights,
        num_samples=len(weights),
        replacement=True
    )

    # --- Create DataLoaders ---
    train_loader = DataLoader(
        train_ds_full, batch_size=BATCH_SIZE,
        sampler=sampler, num_workers=0
    )
    val_loader = DataLoader(
        val_ds_full, batch_size=BATCH_SIZE,
        shuffle=False, num_workers=0
    )

    return train_loader, val_loader, full_ds


# ================================ TRAINING ================================
def load_model(checkpoint_name="best_adenseunet_boosted_fixed.pth"):
    model = ADenseUNet_Boosted(in_channels=3, out_channels=1).to(DEVICE)
    checkpoint_path = os.path.join(CHECKPOINT_DIR, checkpoint_name)

    if os.path.exists(checkpoint_path):
        print(f"\nLoading from checkpoint: {checkpoint_name}")
        checkpoint = torch.load(checkpoint_path, map_location=DEVICE, weights_only=False)
        state_dict = checkpoint["model_state_dict"]
        remapped = {}
        for key, value in state_dict.items():
            if key.startswith("up_final."):
                remapped["final_up." + key.split(".", 1)[1]] = value
            elif key.startswith("final_up."):
                remapped["up_final." + key.split(".", 1)[1]] = value
            else:
                remapped[key] = value
        try:
            model.load_state_dict(state_dict, strict=True)
        except Exception:
            model.load_state_dict(remapped, strict=True)
        start_epoch = checkpoint.get("epoch", 0)
        best_dice = checkpoint.get("best_dice", 0.0)
        print(f"Resuming from epoch {start_epoch}, previous best Dice: {best_dice:.4f}")
    else:
        print("\nTraining from scratch.")
        start_epoch = 0
        best_dice = 0.0

    return model, start_epoch, best_dice


def evaluate_thresholds(model, val_loader, thresholds):
    model.eval()
    threshold_results = defaultdict(dict)

    with torch.no_grad():
        # First collect all predictions and targets
        all_probs = []
        all_targets = []
        all_dataset_names = []
        for images, masks, ds_names in tqdm(val_loader, desc="Collecting predictions"):
            images = images.to(DEVICE)
            logits = model(images)
            probs = torch.sigmoid(logits)
            all_probs.append(probs.cpu())
            all_targets.append(masks)
            all_dataset_names.extend(ds_names)

        all_probs = torch.cat(all_probs, dim=0)
        all_targets = torch.cat(all_targets, dim=0)

        # Now evaluate each threshold
        print("\nEvaluating thresholds...")
        best_threshold = 0.5
        best_overall_dice = 0.0
        overall_metrics = {}
        per_dataset_metrics = defaultdict(dict)

        for threshold in thresholds:
            print(f"\nThreshold: {threshold:.2f}")
            total_metrics = defaultdict(float)
            dataset_metrics = defaultdict(lambda: defaultdict(float))
            dataset_counts = defaultdict(int)

            for i in range(len(all_probs)):
                prob = all_probs[i:i+1]
                target = all_targets[i:i+1]
                ds_name = all_dataset_names[i]

                # Compute metrics for this threshold
                metrics = compute_segmentation_metrics(prob, target, threshold=threshold)
                for k, v in metrics.items():
                    total_metrics[k] += v
                    dataset_metrics[ds_name][k] += v
                dataset_counts[ds_name] += 1

            # Average total metrics
            avg_total = {k: v / len(all_probs) for k, v in total_metrics.items()}
            threshold_results[threshold]["overall"] = avg_total
            threshold_results[threshold]["per_dataset"] = {}
            for ds_name, counts in dataset_counts.items():
                threshold_results[threshold]["per_dataset"][ds_name] = {
                    k: v / counts for k, v in dataset_metrics[ds_name].items()
                }

            print(f"  Overall Dice: {avg_total['dice']:.4f}")
            if avg_total["dice"] > best_overall_dice:
                best_overall_dice = avg_total["dice"]
                best_threshold = threshold

        # Store best threshold metrics
        threshold_results["best_threshold"] = best_threshold
        threshold_results["best_dice"] = best_overall_dice

        return threshold_results


def train_one_epoch(model, train_loader, optimizer, criterion, epoch):
    model.train()
    train_loss = 0.0

    pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{NUM_EPOCHS} [Train]")
    for images, masks, _ in pbar:
        images, masks = images.to(DEVICE), masks.to(DEVICE)
        optimizer.zero_grad()
        logits = model(images)
        loss = criterion(logits, masks)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        train_loss += loss.item() * images.size(0)
        pbar.set_postfix({"Loss": f"{loss.item():.4f}"})

    return train_loss / len(train_loader.dataset)


def validate(model, val_loader, criterion):
    model.eval()
    val_loss = 0.0

    # We'll also collect per-dataset metrics
    per_dataset_totals = defaultdict(lambda: defaultdict(float))
    per_dataset_counts = defaultdict(int)

    with torch.no_grad():
        pbar = tqdm(val_loader, desc="Validating")
        for images, masks, ds_names in pbar:
            images, masks = images.to(DEVICE), masks.to(DEVICE)
            logits = model(images)
            loss = criterion(logits, masks)
            val_loss += loss.item() * images.size(0)

            # Compute per-image metrics
            for i in range(len(ds_names)):
                ds_name = ds_names[i]
                metrics = compute_segmentation_metrics(logits[i:i+1], masks[i:i+1])
                for k, v in metrics.items():
                    per_dataset_totals[ds_name][k] += v
                per_dataset_counts[ds_name] += 1

    # Average val loss and per-dataset metrics
    avg_val_loss = val_loss / len(val_loader.dataset)
    avg_per_dataset = {}
    for ds_name, counts in per_dataset_counts.items():
        avg_per_dataset[ds_name] = {
            k: v / counts for k, v in per_dataset_totals[ds_name].items()
        }

    return avg_val_loss, avg_per_dataset


def main():
    # --- Get data ---
    train_loader, val_loader, full_ds = get_data_loaders()

    # --- Load model and setup ---
    model, start_epoch, best_dice = load_model()

    # --- Loss, optimizer, scheduler ---
    criterion = CombinedLoss(
        ft_alpha=FT_ALPHA,
        ft_beta=FT_BETA,
        ft_gamma=FT_GAMMA,
        lambda_ft=LAMBDA_FT,
        lambda_boundary=LAMBDA_BOUNDARY
    )

    optimizer = optim.AdamW(model.parameters(), lr=INITIAL_LR, weight_decay=1e-4)

    # Cosine annealing from initial LR to final LR
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=NUM_EPOCHS,
        eta_min=FINAL_LR
    )

    # --- Training history ---
    history = {
        "train_loss": [],
        "val_loss": [],
        "val_dice": [],
        "best_dice": best_dice,
        "best_threshold": 0.5,
        "per_dataset_history": []
    }

    # --- Load history if exists ---
    history_path = os.path.join(LOG_DIR, "training_history.json")
    if os.path.exists(history_path):
        with open(history_path, "r") as f:
            history = json.load(f)

    # --- Training loop ---
    for epoch in range(start_epoch, NUM_EPOCHS):
        # Train
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, epoch)
        scheduler.step()

        # Validate
        val_loss, val_per_dataset = validate(model, val_loader, criterion)

        # Get overall Dice from per-dataset
        total_dice = 0
        count = 0
        for ds in val_per_dataset:
            total_dice += val_per_dataset[ds]["dice"]
            count += 1
        avg_dice = total_dice / count

        # Update history
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["val_dice"].append(avg_dice)
        history["per_dataset_history"].append(val_per_dataset)

        # Print summary
        print(f"\nEpoch {epoch+1}/{NUM_EPOCHS}:")
        print(f"  Train Loss: {train_loss:.4f}")
        print(f"  Val Loss:   {val_loss:.4f}")
        print(f"  Val Dice:   {avg_dice:.4f}")
        print("\n--- Per-Dataset Metrics ---")
        for ds_name, metrics in sorted(val_per_dataset.items()):
            print(f"  {ds_name:15s} - Dice: {metrics['dice']:.4f}, IoU: {metrics['iou']:.4f}, Prec: {metrics['precision']:.4f}, Rec: {metrics['recall']:.4f}")

        # Save checkpoint
        checkpoint_data = {
            "epoch": epoch + 1,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "best_dice": best_dice,
            "best_threshold": history["best_threshold"],
        }

        # Save periodic checkpoint every 50 epochs
        if (epoch + 1) % 50 == 0:
            torch.save(checkpoint_data, os.path.join(CHECKPOINT_DIR, f"checkpoint_combined_epoch_{epoch+1}.pth"))
            print(f"\nSaved periodic checkpoint at epoch {epoch+1}")

        # Save best model
        if avg_dice > best_dice:
            best_dice = avg_dice
            history["best_dice"] = best_dice

            # Evaluate thresholds to find best for this best model
            print("\nEvaluating thresholds for best model...")
            threshold_results = evaluate_thresholds(model, val_loader, EVAL_THRESHOLDS)
            history["best_threshold"] = threshold_results["best_threshold"]
            checkpoint_data["best_threshold"] = history["best_threshold"]
            checkpoint_data["best_dice"] = best_dice
            torch.save(checkpoint_data, os.path.join(CHECKPOINT_DIR, "best_adenseunet_combined_improved.pth"))

            print("\n--- Threshold Evaluation Results ---")
            for t in EVAL_THRESHOLDS:
                m = threshold_results[t]["overall"]
                print(f"  Threshold {t:.2f} - Dice: {m['dice']:.4f}, IoU: {m['iou']:.4f}, Prec: {m['precision']:.4f}, Rec: {m['recall']:.4f}")
            print(f"\n  BEST THRESHOLD: {threshold_results['best_threshold']:.2f} (Dice: {threshold_results['best_dice']:.4f})")

            print("\n--- Per-Dataset Metrics (Best Threshold) ---")
            best_thresh = threshold_results["best_threshold"]
            for ds_name, metrics in sorted(threshold_results[best_thresh]["per_dataset"].items()):
                print(f"  {ds_name:15s} - Dice: {metrics['dice']:.4f}, IoU: {metrics['iou']:.4f}, Prec: {metrics['precision']:.4f}, Rec: {metrics['recall']:.4f}")

            print(f"\nSAVED NEW BEST MODEL (Dice: {best_dice:.4f})")

        # Save history
        with open(history_path, "w") as f:
            json.dump(history, f, indent=2)

        print("\n" + "="*120)

    print("\nTRAINING COMPLETE!")
    print(f"Best Dice: {best_dice:.4f} at threshold {history['best_threshold']:.2f}")


if __name__ == "__main__":
    main()
