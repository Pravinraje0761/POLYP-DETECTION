
"""
Evaluation script for Ultimate A-DenseUNet
"""
import os
import torch
import numpy as np
from PIL import Image
import albumentations as A
from albumentations.pytorch import ToTensorV2
from tqdm import tqdm

# Import our model
from ultimate_adenseunet import UltimateADenseUNet

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
IMAGE_SIZE = 512
CHECKPOINT_PATH = "checkpoints/best_adenseunet_ultimate_93_plus.pth"
DATA_ROOT = "archive (1)/PNG"

def dice_score(preds, targets, smooth=1e-6):
    preds = torch.sigmoid(preds)
    preds = preds.view(-1)
    targets = targets.view(-1)
    intersection = (preds * targets).sum()
    return (2. * intersection + smooth) / (preds.sum() + targets.sum() + smooth)

# Load model
print("Loading model...")
model = UltimateADenseUNet(in_channels=3, out_channels=1).to(DEVICE)
if os.path.exists(CHECKPOINT_PATH):
    checkpoint = torch.load(CHECKPOINT_PATH, map_location=DEVICE, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    print(f"Loaded checkpoint from epoch {checkpoint['epoch']} with Dice: {checkpoint['best_dice']:.4f}")
else:
    print("Checkpoint not found!")
    exit()
model.eval()

# Dataset
class PolypDataset:
    def __init__(self, data_root):
        self.data_root = data_root
        self.original_dir = os.path.join(data_root, "Original")
        self.gt_dir = os.path.join(data_root, "Ground Truth")
        self.image_files = sorted([f for f in os.listdir(self.original_dir) if f.endswith('.png')])
        np.random.seed(42)
        np.random.shuffle(self.image_files)
        self.val_files = self.image_files[:int(0.2 * len(self.image_files))]
        
    def __len__(self):
        return len(self.val_files)
        
    def __getitem__(self, idx):
        img_file = self.val_files[idx]
        img_path = os.path.join(self.original_dir, img_file)
        mask_path = os.path.join(self.gt_dir, img_file)
        image = np.array(Image.open(img_path).convert("RGB"))
        mask = np.array(Image.open(mask_path).convert("L"), dtype=np.float32)
        mask = (mask > 127.5).astype(np.float32)
        transform = A.Compose([
            A.Resize(IMAGE_SIZE, IMAGE_SIZE),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2(),
        ])
        augmented = transform(image=image, mask=mask)
        return augmented["image"], augmented["mask"].unsqueeze(0), img_file

dataset = PolypDataset(DATA_ROOT)
total_dice = 0.0

print("\nEvaluating...")
with torch.no_grad():
    for idx in tqdm(range(len(dataset)), desc="Evaluating"):
        img, mask, _ = dataset[idx]
        img = img.unsqueeze(0).to(DEVICE)
        mask = mask.unsqueeze(0).to(DEVICE)
        outputs, _ = model(img)
        dice = dice_score(outputs, mask)
        total_dice += dice.item()

avg_dice = total_dice / len(dataset)
print("\n" + "="*50)
print(f"Average Validation Dice Score: {avg_dice:.4f}")
print("="*50)

