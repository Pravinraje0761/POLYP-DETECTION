
import os
import json
import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from torch.utils.data import DataLoader, Subset, Dataset
import albumentations as A
from albumentations.pytorch import ToTensorV2
import segmentation_models_pytorch as smp
from polypgate.models.adenseunet import ADenseUNet_Boosted


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
IMAGE_SIZE = 320
DATA_ROOT = "combined_dataset"
SPLIT_PATH = "checkpoints/ultimate_split_indices.json"
BEST_UNETPP = "checkpoints/best_pretrained_unetpp.pth"
OUTPUT_DIR = "qualitative_results"
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
        return image, mask, idx, img_path


def get_transforms():
    return A.Compose([
        A.Resize(IMAGE_SIZE, IMAGE_SIZE),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2(),
    ])


def denormalize(tensor):
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    tensor = tensor.permute(1, 2, 0).numpy()
    tensor = std * tensor + mean
    tensor = np.clip(tensor, 0, 1)
    return tensor


def compute_dice_np(mask, pred):
    mask = mask.flatten()
    pred = pred.flatten()
    intersection = (mask * pred).sum()
    return (2. * intersection + 1e-6) / (mask.sum() + pred.sum() + 1e-6)


def main():
    with open(SPLIT_PATH, "r") as f:
        split = json.load(f)

    test_ds = Subset(PolypDataset(DATA_ROOT, get_transforms()), split["test"])
    test_loader = DataLoader(test_ds, batch_size=1, shuffle=False, num_workers=0)

    # load model
    ckpt = torch.load(BEST_UNETPP, map_location=DEVICE, weights_only=False)
    model = smp.UnetPlusPlus(encoder_name=ckpt["encoder"], encoder_weights=None, in_channels=3, classes=1).to(DEVICE)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    # collect a few examples
    examples = []
    with torch.no_grad():
        for i, (img, mask, idx, img_path) in enumerate(test_loader):
            img = img.to(DEVICE)
            pred_prob = torch.sigmoid(model(img))
            pred = (pred_prob > 0.5).float().cpu().numpy()[0, 0]
            mask_np = mask.numpy()[0]
            dice = compute_dice_np(mask_np, pred)
            img_np = denormalize(img.cpu()[0])

            examples.append({
                "img": img_np, "mask": mask_np, "pred": pred,
                "dice": dice, "idx": idx.item(), "path": img_path[0]
            })
            if len(examples) >= 10:
                break

    # sort examples by dice to pick good, medium, bad
    examples_sorted = sorted(examples, key=lambda x: x["dice"], reverse=True)

    # pick top 3 (good), middle 3 (okay), bottom 4 (bad/over/under segmented)
    selected = examples_sorted[:3] + examples_sorted[len(examples_sorted)//2 -1 : len(examples_sorted)//2 +2] + examples_sorted[-4:]

    for i, ex in enumerate(selected):
        fig, axes = plt.subplots(1, 4, figsize=(16, 8))
        axes[0].imshow(ex["img"])
        axes[0].set_title("Original")
        axes[0].axis("off")

        axes[1].imshow(ex["mask"], cmap="gray")
        axes[1].set_title("Ground Truth")
        axes[1].axis("off")

        axes[2].imshow(ex["pred"], cmap="gray")
        axes[2].set_title(f"Prediction (Dice: {ex['dice']:.3f})")
        axes[2].axis("off")

        # overlay
        overlay = ex["img"].copy()
        overlay[ex["mask"] == 1] = [1, 0, 0]  # red for GT
        overlay[ex["pred"] == 1] = [0, 1, 0]  # green for pred
        overlay[(ex["mask"] == 1) & (ex["pred"] == 1)] = [0, 0, 1]  # blue for intersection
        axes[3].imshow(overlay)
        axes[3].set_title("Overlay (GT: red, Pred: green, Intersect: blue)")
        axes[3].axis("off")

        fig.suptitle(f"Example {i+1}")
        fig.tight_layout()
        fig.savefig(os.path.join(OUTPUT_DIR, f"qualitative_example_{i+1}.png"), dpi=150)
        plt.close(fig)

    print(f"Generated {len(selected)} qualitative examples in {OUTPUT_DIR}!")


if __name__ == "__main__":
    main()
