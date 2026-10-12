
import os
import torch
import numpy as np
import albumentations as A
from albumentations.pytorch import ToTensorV2
from PIL import Image
from torch.utils.data import Dataset, DataLoader, random_split
from tqdm import tqdm
from collections import defaultdict

from polypgate.models.adenseunet import ADenseUNet_Boosted
from polypgate.metrics import compute_segmentation_metrics

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
IMAGE_SIZE = 224
BATCH_SIZE = 8
DATA_ROOT = "combined_dataset"
CHECKPOINT_DIR = "checkpoints"

class CombinedDataset(Dataset):
    def __init__(self, data_root, transform=None):
        self.images_dir = os.path.join(data_root, "images")
        self.masks_dir = os.path.join(data_root, "masks")
        self.transform = transform
        self.files = sorted([f for f in os.listdir(self.images_dir) if f.lower().endswith((".png", ".jpg", ".jpeg"))])

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        name = self.files[idx]
        image = np.array(Image.open(os.path.join(self.images_dir, name)).convert("RGB"))
        mask = np.array(Image.open(os.path.join(self.masks_dir, name)).convert("L"), dtype=np.float32)
        mask = (mask > 127.5).astype(np.float32)
        if self.transform:
            augmented = self.transform(image=image, mask=mask)
            return augmented["image"], augmented["mask"].unsqueeze(0), name
        return image, mask, name

def load_model(checkpoint_name):
    checkpoint_path = os.path.join(CHECKPOINT_DIR, checkpoint_name)
    model = ADenseUNet_Boosted(in_channels=3, out_channels=1).to(DEVICE)
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
        try:
            model.load_state_dict(remapped, strict=True)
        except Exception as e:
            print(f"Failed to load {checkpoint_name}: {str(e)}")
            return None

    model.eval()
    return model, checkpoint

def evaluate_model(model, dataset):
    transform = A.Compose([
        A.Resize(IMAGE_SIZE, IMAGE_SIZE),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2(),
    ])

    class TransformedDataset(Dataset):
        def __init__(self, base_dataset, transform):
            self.base_dataset = base_dataset
            self.transform = transform

        def __len__(self):
            return len(self.base_dataset)

        def __getitem__(self, idx):
            original_idx = self.base_dataset.indices[idx] if hasattr(self.base_dataset, "indices") else idx
            name = self.base_dataset.dataset.files[original_idx] if hasattr(self.base_dataset, "indices") else self.base_dataset.files[idx]
            image = np.array(Image.open(os.path.join(self.base_dataset.dataset.images_dir if hasattr(self.base_dataset, "indices") else self.base_dataset.images_dir, name)).convert("RGB"))
            mask = np.array(Image.open(os.path.join(self.base_dataset.dataset.masks_dir if hasattr(self.base_dataset, "indices") else self.base_dataset.masks_dir, name)).convert("L"), dtype=np.float32)
            mask = (mask > 127.5).astype(np.float32)
            augmented = self.transform(image=image, mask=mask)
            return augmented["image"], augmented["mask"].unsqueeze(0), name

    transformed_ds = TransformedDataset(dataset, transform)
    loader = DataLoader(transformed_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    totals = defaultdict(float)
    count = 0
    with torch.no_grad():
        for images, masks, _ in tqdm(loader, desc="Evaluating"):
            images = images.to(DEVICE)
            masks = masks.to(DEVICE)
            logits = model(images)
            for i in range(images.size(0)):
                metrics = compute_segmentation_metrics(logits[i:i+1], masks[i:i+1])
                for k, v in metrics.items():
                    totals[k] += v
                count += 1
    return {k: v/count for k, v in totals.items()}

def main():
    print("="*120)
    print("Evaluating COMBINED DATASET checkpoints on validation split:")
    print("="*120)

    full_ds = CombinedDataset(DATA_ROOT)
    train_size = int(0.8 * len(full_ds))
    val_size = len(full_ds) - train_size
    _, val_ds = random_split(full_ds, [train_size, val_size], generator=torch.Generator().manual_seed(42))

    # Only check combined dataset checkpoints
    combined_checkpoints = sorted([
        f for f in os.listdir(CHECKPOINT_DIR) 
        if f.startswith("checkpoint_combined") or f.startswith("best_adenseunet_combined")
    ])

    best_accuracy = 0
    best_checkpoint = ""
    best_metrics = None

    for filename in combined_checkpoints:
        print(f"\n--- {filename} ---")
        model_load_result = load_model(filename)
        if model_load_result is None:
            continue
        model, checkpoint = model_load_result
        metrics = evaluate_model(model, val_ds)
        
        epoch = checkpoint.get("epoch", "N/A")
        best_dice_training = checkpoint.get("best_dice", "N/A")
        
        print(f"  Training Epoch:  {epoch}")
        print(f"  Training Best Dice: {best_dice_training:.4f} ({best_dice_training*100:.2f}%)" if isinstance(best_dice_training, float) else f"  Training Best Dice: {best_dice_training}")
        print(f"  Validation Accuracy:  {metrics['accuracy']:.4f} ({metrics['accuracy']*100:.2f}%)")
        print(f"  Validation Dice:      {metrics['dice']:.4f} ({metrics['dice']*100:.2f}%)")
        print(f"  Validation IoU:       {metrics['iou']:.4f} ({metrics['iou']*100:.2f}%)")
        print(f"  Validation Precision: {metrics['precision']:.4f} ({metrics['precision']*100:.2f}%)")
        print(f"  Validation Recall:    {metrics['recall']:.4f} ({metrics['recall']*100:.2f}%)")

        if metrics['accuracy'] > best_accuracy:
            best_accuracy = metrics['accuracy']
            best_checkpoint = filename
            best_metrics = metrics

    print("\n" + "="*120)
    print(f"BEST CHECKPOINT: {best_checkpoint}")
    print(f"BEST ACCURACY:   {best_accuracy:.4f} ({best_accuracy*100:.2f}%)")
    if best_metrics:
        print("ALL METRICS FOR BEST CHECKPOINT:")
        for k, v in best_metrics.items():
            print(f"   - {k:12s}: {v:.4f} ({v*100:.2f}%)")
    print("="*120)

if __name__ == "__main__":
    main()
