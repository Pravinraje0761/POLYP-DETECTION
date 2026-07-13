"""
Metrics module for polyp detection.
Includes classification, segmentation, and explainability metrics.
"""
import torch
import numpy as np
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
    confusion_matrix, roc_curve, auc
)
import matplotlib.pyplot as plt
import seaborn as sns


# ==================== Classification Metrics ====================
def compute_classification_metrics(y_true, y_pred, y_prob):
    """
    Compute all classification metrics.
    
    Args:
        y_true (np.array): True labels
        y_pred (np.array): Predicted labels
        y_prob (np.array): Predicted probabilities (for positive class)
        
    Returns:
        dict: Dictionary of metrics
    """
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    f1 = f1_score(y_true, y_pred, zero_division=0)
    roc_auc = roc_auc_score(y_true, y_prob)
    
    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "specificity": specificity,
        "f1": f1,
        "roc_auc": roc_auc,
        "confusion_matrix": np.array([[tn, fp], [fn, tp]])
    }


# ==================== Segmentation Metrics ====================
def dice_score(pred, target, smooth=1e-6):
    """
    Compute Dice Score.
    """
    pred = (pred > 0.5).float()
    pred = pred.view(-1)
    target = target.view(-1)
    intersection = (pred * target).sum()
    return (2. * intersection + smooth) / (pred.sum() + target.sum() + smooth)


def iou_score(pred, target, smooth=1e-6):
    """
    Compute IoU (Jaccard Index).
    """
    pred = (pred > 0.5).float()
    pred = pred.view(-1)
    target = target.view(-1)
    intersection = (pred * target).sum()
    union = pred.sum() + target.sum() - intersection
    return (intersection + smooth) / (union + smooth)


def segmentation_precision(pred, target, smooth=1e-6):
    """Pixel-level precision for binary segmentation."""
    pred = (pred > 0.5).float().view(-1)
    target = target.view(-1)
    tp = (pred * target).sum()
    return (tp + smooth) / (pred.sum() + smooth)


def segmentation_recall(pred, target, smooth=1e-6):
    """Pixel-level recall / sensitivity for binary segmentation."""
    pred = (pred > 0.5).float().view(-1)
    target = target.view(-1)
    tp = (pred * target).sum()
    return (tp + smooth) / (target.sum() + smooth)


def segmentation_specificity(pred, target, smooth=1e-6):
    """Pixel-level specificity for binary segmentation."""
    pred = (pred > 0.5).float().view(-1)
    target = target.view(-1)
    tn = ((1 - pred) * (1 - target)).sum()
    fp = (pred * (1 - target)).sum()
    return (tn + smooth) / (tn + fp + smooth)


def segmentation_accuracy(pred, target, smooth=1e-6):
    """Pixel-level segmentation accuracy."""
    pred = (pred > 0.5).float().view(-1)
    target = target.view(-1)
    correct = (pred == target).sum()
    return (correct + smooth) / (target.numel() + smooth)


def compute_segmentation_metrics(pred, target, threshold=0.5, smooth=1e-6):
    """
    Compute all standard binary segmentation metrics.

    Args:
        pred: Predicted probabilities or logits (torch.Tensor)
        target: Ground truth mask (torch.Tensor, values 0/1)
        threshold: Binarization threshold for predictions

    Returns:
        dict with dice, iou, precision, recall, specificity, accuracy, f1
    """
    if pred.dtype in (torch.float32, torch.float16, torch.float64):
        if pred.max() > 1.0 or pred.min() < 0.0:
            pred = torch.sigmoid(pred)
        pred_bin = (pred > threshold).float()
    else:
        pred_bin = pred.float()

    target = target.float()
    pred_flat = pred_bin.view(-1)
    target_flat = target.view(-1)

    tp = (pred_flat * target_flat).sum()
    fp = (pred_flat * (1 - target_flat)).sum()
    fn = ((1 - pred_flat) * target_flat).sum()
    tn = ((1 - pred_flat) * (1 - target_flat)).sum()

    dice = (2 * tp + smooth) / (2 * tp + fp + fn + smooth)
    iou = (tp + smooth) / (tp + fp + fn + smooth)
    precision = (tp + smooth) / (tp + fp + smooth)
    recall = (tp + smooth) / (tp + fn + smooth)
    specificity = (tn + smooth) / (tn + fp + smooth)
    accuracy = (tp + tn + smooth) / (tp + tn + fp + fn + smooth)
    f1 = (2 * precision * recall + smooth) / (precision + recall + smooth)

    return {
        "dice": float(dice),
        "iou": float(iou),
        "precision": float(precision),
        "recall": float(recall),
        "specificity": float(specificity),
        "accuracy": float(accuracy),
        "f1": float(f1),
    }


# ==================== Explainability Metrics ====================
def attention_concentration_ratio(heatmap, mask):
    """
    Stage 10: Attention Concentration Ratio (ACR).
    ACR = Heatmap energy inside mask / Total heatmap energy
    
    Args:
        heatmap (np.array): Grad-CAM/attention heatmap
        mask (np.array): Ground truth or predicted mask
        
    Returns:
        float: ACR value
    """
    # Normalize heatmap
    heatmap = heatmap.astype(np.float32)
    heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min() + 1e-8)
    
    # Resize mask to match heatmap
    if heatmap.shape != mask.shape:
        import cv2
        mask = cv2.resize(mask, (heatmap.shape[1], heatmap.shape[0]))
    
    mask = (mask > 0.5).astype(np.float32)
    
    # Compute energies
    energy_inside = np.sum(heatmap * mask)
    energy_total = np.sum(heatmap)
    
    return energy_inside / (energy_total + 1e-8)


def gradcam_iou(heatmap, mask, threshold=0.5):
    """
    Stage 10: Grad-CAM IoU.
    Overlap between binarized heatmap and ground truth mask.
    
    Args:
        heatmap (np.array): Grad-CAM heatmap
        mask (np.array): Ground truth mask
        threshold (float): Threshold for binarizing heatmap
        
    Returns:
        float: IoU value
    """
    # Normalize and binarize heatmap
    heatmap = heatmap.astype(np.float32)
    heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min() + 1e-8)
    heatmap_binary = (heatmap > threshold).astype(np.float32)
    
    # Resize mask if needed
    if heatmap.shape != mask.shape:
        import cv2
        mask = cv2.resize(mask, (heatmap.shape[1], heatmap.shape[0]))
    mask_binary = (mask > 0.5).astype(np.float32)
    
    # Compute IoU
    intersection = np.sum(heatmap_binary * mask_binary)
    union = np.sum(heatmap_binary) + np.sum(mask_binary) - intersection
    
    return intersection / (union + 1e-8)


def pointing_game_accuracy(heatmap, mask):
    """
    Stage 10: Pointing Game Accuracy.
    Check whether maximum activation lies inside the polyp.
    
    Args:
        heatmap (np.array): Grad-CAM/attention heatmap
        mask (np.array): Ground truth mask
        
    Returns:
        bool: True if max activation is inside mask
    """
    # Resize mask if needed
    if heatmap.shape != mask.shape:
        import cv2
        mask = cv2.resize(mask, (heatmap.shape[1], heatmap.shape[0]))
    mask_binary = (mask > 0.5).astype(np.float32)
    
    # Find position of max activation
    y, x = np.unravel_index(np.argmax(heatmap), heatmap.shape)
    
    return mask_binary[y, x] > 0.5


# ==================== Visualization ====================
def plot_roc_curve(y_true, y_prob, save_path=None):
    """
    Plot ROC Curve.
    """
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    roc_auc = auc(fpr, tpr)
    
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic')
    plt.legend(loc="lower right")
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()


def plot_confusion_matrix(cm, save_path=None, classes=['Normal', 'Polyp']):
    """
    Plot Confusion Matrix.
    """
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=classes, yticklabels=classes)
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title('Confusion Matrix')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()


def plot_training_curves(train_losses, val_losses, train_accs, val_accs, save_path=None):
    """
    Plot training and validation curves.
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Loss curve
    ax1.plot(train_losses, label='Train Loss')
    ax1.plot(val_losses, label='Val Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('Loss Curve')
    ax1.legend()
    
    # Accuracy curve
    ax2.plot(train_accs, label='Train Accuracy')
    ax2.plot(val_accs, label='Val Accuracy')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy')
    ax2.set_title('Accuracy Curve')
    ax2.legend()
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
