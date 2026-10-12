import os
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import albumentations as A
from albumentations.pytorch import ToTensorV2
import numpy as np


class PolypDataset(Dataset):
    def __init__(self, image_dir, mask_dir=None, transform=None):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.transform = transform
        self.images = os.listdir(image_dir)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = os.path.join(self.image_dir, self.images[idx])
        image = np.array(Image.open(img_path).convert("RGB"))
        
        if self.mask_dir:
            mask_path = os.path.join(self.mask_dir, self.images[idx])
            mask = np.array(Image.open(mask_path).convert("L"), dtype=np.float32)
            mask = (mask > 127.5).astype(np.float32)
            
            if self.transform:
                augmented = self.transform(image=image, mask=mask)
                image = augmented["image"]
                mask = augmented["mask"]
                mask = mask.unsqueeze(0)
            return image, mask
        else:
            if self.transform:
                augmented = self.transform(image=image)
                image = augmented["image"]
            return image


def get_transforms(image_size=224):
    train_transform = A.Compose([
        A.Resize(height=image_size, width=image_size),
        A.RandomRotate90(),
        A.Flip(p=0.5),
        A.RandomBrightnessContrast(p=0.2),
        A.Normalize(mean=[0.0, 0.0, 0.0], std=[1.0, 1.0, 1.0]),
        ToTensorV2(),
    ])
    
    val_transform = A.Compose([
        A.Resize(height=image_size, width=image_size),
        A.Normalize(mean=[0.0, 0.0, 0.0], std=[1.0, 1.0, 1.0]),
        ToTensorV2(),
    ])
    
    return train_transform, val_transform


def get_dataloaders(train_img_dir, train_mask_dir, val_img_dir, val_mask_dir, 
                    batch_size=16, image_size=224):
    train_transform, val_transform = get_transforms(image_size)
    
    train_dataset = PolypDataset(train_img_dir, train_mask_dir, train_transform)
    val_dataset = PolypDataset(val_img_dir, val_mask_dir, val_transform)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    
    return train_loader, val_loader
