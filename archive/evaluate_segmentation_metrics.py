"""
Evaluate segmentation model with full metric suite:
Dice (DSC/F1), IoU, Precision, Recall/Sensitivity, Specificity, Accuracy.
"""
import argparse
import os

import albumentations as A
from albumentations.pytorch import ToTensorV2
import numpy as np
from PIL import Image
import torch
from torch.utils.data import DataLoader, Dataset, random_split
from tqdm import tqdm

from polypgate.models.adenseunet import ADenseUNet_Boosted
from polypgate.metrics import compute_segmentation_metrics


def parse_args():
    parser = argparse.ArgumentParser(description="Segmentation metrics evaluation")
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="checkpoints/best_adenseunet_boosted_fixed.pth",
        help="Path to segmentation checkpoint",
    )
    parser.add_argument(
        "--data_root",
        type=str,
        default="combined_dataset",
        help="Dataset root (images/ + masks/)",
    )
    parser.add_argument("--image_size", type=int, default=224)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--val_split", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--split",
        choices=["val", "all"],
        default="val",
        help="Evaluate validation split or full dataset",
    )
    return parser.parse_args()


class SegmentationDataset(Dataset):
    def __init__(self, data_root, transform=None):
        images_dir = os.path.join(data_root, "images")
        masks_dir = os.path.join(data_root, "masks")
        if os.path.isdir(images_dir) and os.path.isdir(masks_dir):
            self.images_dir = images_dir
            self.masks_dir = masks_dir
        else:
            self.images_dir = os.path.join(data_root, "Original")
            self.masks_dir = os.path.join(data_root, "Ground Truth")
        self.transform = transform
        self.files = sorted(
            f for f in os.listdir(self.images_dir)
            if f.lower().endswith((".png", ".jpg", ".jpeg"))
        )

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        name = self.files[idx]
        image = np.array(Image.open(os.path.join(self.images_dir, name)).convert("RGB"))
        mask = np.array(Image.open(os.path.join(self.masks_dir, name)).convert("L"), dtype=np.float32)
        mask = (mask > 127.5).astype(np.float32)
        if self.transform:
            augmented = self.transform(image=image, mask=mask)
            image = augmented["image"]
            mask = augmented["mask"]
        return image, mask.unsqueeze(0), name


def load_model(checkpoint_path, device):
    model = ADenseUNet_Boosted(in_channels=3, out_channels=1).to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    state_dict = checkpoint["model_state_dict"]
    remapped = {}
    for key, value in state_dict.items():
        if key.startswith("up_final."):
            remapped["final_up." + key.split(".", 1)[1]] = value
        else:
            remapped[key] = value
    model.load_state_dict(remapped, strict=True)
    model.eval()
    return model, checkpoint


def get_transform(image_size):
    return A.Compose([
        A.Resize(image_size, image_size),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2(),
    ])


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if not os.path.isdir(os.path.join(args.data_root, "images")) and not os.path.isdir(
        os.path.join(args.data_root, "Original")
    ):
        raise FileNotFoundError(
            f"Dataset not found at {args.data_root} (expected images/masks or Original/Ground Truth)"
        )

    print("=" * 80)
    print("SEGMENTATION METRICS EVALUATION")
    print("=" * 80)
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Dataset:    {args.data_root}")
    print(f"Split:      {args.split}")
    print(f"Device:     {device}")
    print("=" * 80)

    model, checkpoint = load_model(args.checkpoint, device)
    print(f"Loaded checkpoint Dice (training record): {checkpoint.get('best_dice', 'N/A')}")

    full_dataset = SegmentationDataset(args.data_root, transform=get_transform(args.image_size))
    if args.split == "val":
        val_size = int(len(full_dataset) * args.val_split)
        train_size = len(full_dataset) - val_size
        _, eval_dataset = random_split(
            full_dataset,
            [train_size, val_size],
            generator=torch.Generator().manual_seed(args.seed),
        )
    else:
        eval_dataset = full_dataset

    loader = DataLoader(eval_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)

    totals = {
        "dice": 0.0,
        "iou": 0.0,
        "precision": 0.0,
        "recall": 0.0,
        "specificity": 0.0,
        "accuracy": 0.0,
        "f1": 0.0,
    }
    count = 0

    with torch.no_grad():
        for images, masks, _ in tqdm(loader, desc="Evaluating"):
            images = images.to(device)
            masks = masks.to(device)
            logits = model(images)

            for i in range(images.size(0)):
                metrics = compute_segmentation_metrics(logits[i : i + 1], masks[i : i + 1])
                for key in totals:
                    totals[key] += metrics[key]
                count += 1

    averages = {key: value / count for key, value in totals.items()}

    print("\nResults (averaged over samples):")
    print(f"  Samples evaluated: {count}")
    print(f"  Dice (DSC):        {averages['dice']:.4f}  ({averages['dice'] * 100:.2f}%)")
    print(f"  IoU (Jaccard):     {averages['iou']:.4f}  ({averages['iou'] * 100:.2f}%)")
    print(f"  F1:                {averages['f1']:.4f}  ({averages['f1'] * 100:.2f}%)")
    print(f"  Precision:         {averages['precision']:.4f}  ({averages['precision'] * 100:.2f}%)")
    print(f"  Recall/Sensitivity:{averages['recall']:.4f}  ({averages['recall'] * 100:.2f}%)")
    print(f"  Specificity:       {averages['specificity']:.4f}  ({averages['specificity'] * 100:.2f}%)")
    print(f"  Segmentation Acc:  {averages['accuracy']:.4f}  ({averages['accuracy'] * 100:.2f}%)")
    print("=" * 80)


if __name__ == "__main__":
    main()
