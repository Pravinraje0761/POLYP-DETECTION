"""
Dataset module for Hyper-Kvasir polyp detection.
Handles data loading, train/val/test splits, and augmentations.
"""
import os
import pandas as pd
import numpy as np
from PIL import Image
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import Dataset, DataLoader
import albumentations as A
from albumentations.pytorch import ToTensorV2


class PolypDataset(Dataset):
    """
    Dataset class for polyp detection.
    Supports both segmentation (if masks are available) and classification.
    """
    def __init__(self, image_paths, mask_paths=None, labels=None, transform=None, image_size=224):
        """
        Args:
            image_paths (list): List of paths to input images
            mask_paths (list, optional): List of paths to segmentation masks
            labels (list, optional): List of classification labels (0=normal, 1=polyp)
            transform (callable, optional): Transform to apply to images and masks
            image_size (int): Target image size
        """
        self.image_paths = image_paths
        self.mask_paths = mask_paths
        self.labels = labels
        self.transform = transform
        self.image_size = image_size

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        # Load image
        image = np.array(Image.open(self.image_paths[idx]).convert("RGB"))
        
        # Load mask if available
        mask = None
        if self.mask_paths:
            mask = np.array(Image.open(self.mask_paths[idx]).convert("L"))
            mask = (mask > 127.5).astype(np.float32)
        
        # Load label if available
        label = None
        if self.labels:
            label = self.labels[idx]
        
        # Apply transforms
        if self.transform:
            if mask is not None:
                augmented = self.transform(image=image, mask=mask)
                image = augmented["image"]
                mask = augmented["mask"]
            else:
                augmented = self.transform(image=image)
                image = augmented["image"]
        
        # Prepare output dict
        item = {"image": image}
        
        if mask is not None:
            item["mask"] = mask.unsqueeze(0) if torch.is_tensor(mask) else torch.tensor(mask).unsqueeze(0)
        
        if label is not None:
            item["label"] = torch.tensor(label, dtype=torch.long)
        
        return item


def get_transforms(image_size=224, is_train=True):
    """
    Get train/val/test transforms.
    
    Args:
        image_size (int): Target image size
        is_train (bool): Whether it's training set
        
    Returns:
        Albumentations Compose object
    """
    if is_train:
        return A.Compose([
            A.Resize(height=image_size, width=image_size),
            A.RandomRotate90(p=0.5),
            A.HorizontalFlip(p=0.5),
            A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
            A.GaussianBlur(blur_limit=(3, 7), p=0.3),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2(),
        ])
    else:
        return A.Compose([
            A.Resize(height=image_size, width=image_size),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2(),
        ])


def load_hyper_kvasir_data(data_root):
    """
    Load Hyper-Kvasir dataset paths.
    
    Args:
        data_root (str): Path to Hyper-Kvasir dataset root
        
    Returns:
        Tuple of (image_paths, labels)
    """
    polyp_images = []
    normal_images = []
    
    # Load polyp images from pathological-findings/polyps
    polyp_dir = os.path.join(data_root, "lower-gi-tract", "pathological-findings", "polyps")
    for root, dirs, files in os.walk(polyp_dir):
        for file in files:
            if file.lower().endswith((".jpg", ".png")):
                polyp_images.append(os.path.join(root, file))
    
    # Load normal images from anatomical-landmarks
    normal_dir = os.path.join(data_root, "lower-gi-tract", "anatomical-landmarks")
    for root, dirs, files in os.walk(normal_dir):
        for file in files:
            if file.lower().endswith((".jpg", ".png")):
                normal_images.append(os.path.join(root, file))
    
    # Combine and create labels
    image_paths = normal_images + polyp_images
    labels = [0] * len(normal_images) + [1] * len(polyp_images)
    
    print(f"Loaded {len(normal_images)} normal images and {len(polyp_images)} polyp images")
    
    return image_paths, labels


def load_archive_data(data_root):
    """
    Load archive (1) dataset paths.
    
    Args:
        data_root (str): Path to archive dataset root (should contain metadata.csv)
        
    Returns:
        Tuple of (image_paths, mask_paths, labels)
    """
    import pandas as pd
    
    metadata_path = os.path.join(data_root, "metadata.csv")
    df = pd.read_csv(metadata_path)
    
    image_paths = []
    mask_paths = []
    labels = []
    
    for _, row in df.iterrows():
        # Use PNG paths as they are smaller
        img_path = os.path.join(data_root, row["png_image_path"].replace("/", os.sep))
        mask_path = os.path.join(data_root, row["png_mask_path"].replace("/", os.sep))
        
        image_paths.append(img_path)
        mask_paths.append(mask_path)
        labels.append(1)  # All images in archive are polyp images
    
    print(f"Loaded {len(image_paths)} polyp images with masks from archive dataset")
    
    return image_paths, mask_paths, labels


def get_dataloaders(data_root, image_size=224, batch_size=16, random_seed=42):
    """
    Get train/val/test dataloaders.
    
    Args:
        data_root (str): Path to dataset root
        image_size (int): Target image size
        batch_size (int): Batch size
        random_seed (int): Random seed for reproducibility
        
    Returns:
        Tuple of (train_loader, val_loader, test_loader)
    """
    # Load data
    image_paths, labels = load_hyper_kvasir_data(data_root)
    
    # Split: 70% train, 15% val, 15% test
    X_train, X_temp, y_train, y_temp = train_test_split(
        image_paths, labels, test_size=0.3, random_state=random_seed, stratify=labels
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, random_state=random_seed, stratify=y_temp
    )
    
    # Get transforms
    train_transform = get_transforms(image_size, is_train=True)
    val_test_transform = get_transforms(image_size, is_train=False)
    
    # Create datasets
    train_dataset = PolypDataset(X_train, labels=y_train, transform=train_transform, image_size=image_size)
    val_dataset = PolypDataset(X_val, labels=y_val, transform=val_test_transform, image_size=image_size)
    test_dataset = PolypDataset(X_test, labels=y_test, transform=val_test_transform, image_size=image_size)
    
    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=True)
    
    return train_loader, val_loader, test_loader
