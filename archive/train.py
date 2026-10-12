
"""
Stage 6: Training script for polyp detection (UPGRADED).
UPGRADES:
- Cosine Annealing with Warmup
- Gradient Clipping
- Better logging
Includes mixed precision, early stopping, cosine annealing, and complete pipeline integration.
"""
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import GradScaler, autocast
from torch.optim.lr_scheduler import CosineAnnealingLR
import numpy as np
from tqdm import tqdm
import argparse

from polypgate.data.dataset import get_dataloaders
from adenseunet import ADenseUNet, CombinedLoss
from polypgate.models.vit_classifier import PolypViT, FocalLoss
from polypgate.preprocessing.pipeline import complete_preprocessing_pipeline
from polypgate.metrics import (
    compute_classification_metrics, dice_score, iou_score,
    plot_roc_curve, plot_confusion_matrix, plot_training_curves
)


def parse_args():
    parser = argparse.ArgumentParser(description="Polyp Detection Training (UPGRADED)")
    parser.add_argument("--data_root", type=str, 
                        default="hyper-kvasir-labeled-images (1)/labeled-images",
                        help="Path to Hyper-Kvasir dataset root")
    parser.add_argument("--seg_checkpoint", type=str, default="checkpoints/best_adenseunet_ultimate.pth",
                        help="Path to pretrained A-DenseUNet checkpoint")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size")
    parser.add_argument("--image_size", type=int, default=224, help="Image size")
    parser.add_argument("--epochs", type=int, default=120, help="Number of epochs")
    parser.add_argument("--warmup_epochs", type=int, default=10, help="Number of warmup epochs")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--min_lr", type=float, default=1e-6, help="Minimum learning rate for cosine annealing")
    parser.add_argument("--weight_decay", type=float, default=1e-4, help="Weight decay")
    parser.add_argument("--early_stopping", type=int, default=15, help="Early stopping patience")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu",
                        help="Device to use (cuda/cpu)")
    parser.add_argument("--output_dir", type=str, default="outputs", help="Output directory")
    parser.add_argument("--ablation", type=str, default="full", 
                        choices=["seg_only", "rgb", "full"],
                        help="Ablation study mode: seg_only, rgb, full")
    return parser.parse_args()


class CosineAnnealingWithWarmupLR(torch.optim.lr_scheduler._LRScheduler):
    """
    Custom LR scheduler: linear warmup followed by cosine annealing.
    """
    def __init__(self, optimizer, warmup_epochs, total_epochs, min_lr=1e-6, last_epoch=-1):
        self.warmup_epochs = warmup_epochs
        self.total_epochs = total_epochs
        self.min_lr = min_lr
        self.base_lrs = [param_group['lr'] for param_group in optimizer.param_groups]
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        epoch = self.last_epoch + 1
        if epoch < self.warmup_epochs:
            # Linear warmup
            return [base_lr * (epoch + 1) / self.warmup_epochs for base_lr in self.base_lrs]
        else:
            # Cosine annealing
            progress = (epoch - self.warmup_epochs) / (self.total_epochs - self.warmup_epochs)
            cosine_lr = self.min_lr + 0.5 * (self.base_lrs[0] - self.min_lr) * (1 + np.cos(np.pi * progress))
            return [cosine_lr for _ in self.base_lrs]


def main():
    args = parse_args()
    device = torch.device(args.device)
    
    # Create output directories
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, "checkpoints"), exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, "figures"), exist_ok=True)
    
    # Load dataloaders
    print("Loading dataset...")
    train_loader, val_loader, test_loader = get_dataloaders(
        args.data_root, args.image_size, args.batch_size
    )
    
    # Initialize models
    print("Initializing models...")
    # Use upgraded A-DenseUNet
    seg_model = ADenseUNet(in_channels=3, out_channels=1).to(device)
    if args.seg_checkpoint:
        seg_checkpoint = torch.load(args.seg_checkpoint, map_location=device, weights_only=False)
        seg_model.load_state_dict(seg_checkpoint["model_state_dict"], strict=False)
    
    cls_model = PolypViT(num_classes=2, pretrained=True, use_lora=True).to(device)
    
    # Loss, optimizer, scheduler
    # Use upgraded CombinedLoss from adenseunet.py
    cls_criterion = FocalLoss(alpha=0.25, gamma=2.0)
    # Use AdamW with weight decay
    optimizer = optim.AdamW(
        [p for p in cls_model.parameters() if p.requires_grad], 
        lr=args.lr, weight_decay=args.weight_decay
    )
    # Use our custom cosine annealing with warmup scheduler
    scheduler = CosineAnnealingWithWarmupLR(
        optimizer, 
        warmup_epochs=args.warmup_epochs, 
        total_epochs=args.epochs, 
        min_lr=args.min_lr
    )
    # Gradient scaler for mixed precision training
    scaler = GradScaler()
    
    # Early stopping
    best_val_auc = 0.0
    epochs_no_improve = 0
    # Tracking metrics
    train_losses = []
    val_losses = []
    train_accs = []
    val_accs = []
    
    # Training loop
    print("Starting training...")
    for epoch in range(args.epochs):
        # ========== Training Phase ==========
        cls_model.train()
        train_running_loss = 0.0
        train_preds = []
        train_labels = []
        train_probs = []
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs} (Train)")
        for batch in pbar:
            images = batch["image"].to(device)
            labels = batch["label"].to(device)
            
            optimizer.zero_grad()
            
            with autocast():
                # Preprocess: apply our pipeline based on ablation mode
                processed_images = []
                for img in images:
                    # Convert back to numpy for preprocessing
                    img_np = img.permute(1, 2, 0).cpu().numpy()
                    # Denormalize
                    mean = np.array([0.485, 0.456, 0.406])
                    std = np.array([0.229, 0.224, 0.225])
                    img_np = std * img_np + mean
                    img_np = np.clip(img_np, 0, 1) * 255
                    img_np = img_np.astype(np.uint8)
                    
                    if args.ablation == "seg_only":
                        # Only resize and normalize
                        processed = img_np
                    elif args.ablation == "rgb":
                        # Seg + RGB enhancement
                        pp = complete_preprocessing_pipeline(img_np, seg_model, device)
                        processed = pp["blended_rgb"]
                    else:  # full
                        # Complete pipeline
                        pp = complete_preprocessing_pipeline(img_np, seg_model, device)
                        processed = pp["after_clahe"]
                    
                    # Convert back to tensor
                    from albumentations import Compose, Resize, Normalize
                    from albumentations.pytorch import ToTensorV2
                    transform = Compose([
                        Resize(args.image_size, args.image_size),
                        Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                        ToTensorV2()
                    ])
                    processed_tensor = transform(image=processed)["image"]
                    processed_images.append(processed_tensor)
                
                processed_images = torch.stack(processed_images).to(device)
                
                # Forward pass
                logits, _ = cls_model(processed_images)
                loss = cls_criterion(logits, labels)
            
            # Backward pass with mixed precision
            scaler.scale(loss).backward()
            # Gradient clipping
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(cls_model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
            
            # Track metrics
            train_running_loss += loss.item() * images.size(0)
            probs = torch.softmax(logits, dim=1)
            preds = torch.argmax(probs, dim=1)
            
            train_preds.extend(preds.detach().cpu().numpy())
            train_labels.extend(labels.detach().cpu().numpy())
            train_probs.extend(probs[:, 1].detach().cpu().numpy())
            
            pbar.set_postfix({"loss": loss.item(), "lr": optimizer.param_groups[0]["lr"]})
        
        # Update scheduler
        scheduler.step()
        
        # Compute train metrics
        train_loss = train_running_loss / len(train_loader.dataset)
        train_metrics = compute_classification_metrics(
            np.array(train_labels), np.array(train_preds), np.array(train_probs)
        )
        
        # ========== Validation Phase ==========
        cls_model.eval()
        val_running_loss = 0.0
        val_preds = []
        val_labels = []
        val_probs = []
        
        with torch.no_grad():
            pbar = tqdm(val_loader, desc=f"Epoch {epoch+1}/{args.epochs} (Val)")
            for batch in pbar:
                images = batch["image"].to(device)
                labels = batch["label"].to(device)
                
                with autocast():
                    # Preprocess
                    processed_images = []
                    for img in images:
                        img_np = img.permute(1, 2, 0).cpu().numpy()
                        mean = np.array([0.485, 0.456, 0.406])
                        std = np.array([0.229, 0.224, 0.225])
                        img_np = std * img_np + mean
                        img_np = np.clip(img_np, 0, 1) * 255
                        img_np = img_np.astype(np.uint8)
                        
                        if args.ablation == "seg_only":
                            processed = img_np
                        elif args.ablation == "rgb":
                            pp = complete_preprocessing_pipeline(img_np, seg_model, device)
                            processed = pp["blended_rgb"]
                        else:
                            pp = complete_preprocessing_pipeline(img_np, seg_model, device)
                            processed = pp["after_clahe"]
                        
                        from albumentations import Compose, Resize, Normalize
                        from albumentations.pytorch import ToTensorV2
                        transform = Compose([
                            Resize(args.image_size, args.image_size),
                            Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                            ToTensorV2()
                        ])
                        processed_tensor = transform(image=processed)["image"]
                        processed_images.append(processed_tensor)
                    
                    processed_images = torch.stack(processed_images).to(device)
                    logits, _ = cls_model(processed_images)
                    loss = cls_criterion(logits, labels)
                
                val_running_loss += loss.item() * images.size(0)
                probs = torch.softmax(logits, dim=1)
                preds = torch.argmax(probs, dim=1)
                
                val_preds.extend(preds.detach().cpu().numpy())
                val_labels.extend(labels.detach().cpu().numpy())
                val_probs.extend(probs[:, 1].detach().cpu().numpy())
                
                pbar.set_postfix({"loss": loss.item()})
        
        val_loss = val_running_loss / len(val_loader.dataset)
        val_metrics = compute_classification_metrics(
            np.array(val_labels), np.array(val_preds), np.array(val_probs)
        )
        
        # Track for plotting
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_accs.append(train_metrics["accuracy"])
        val_accs.append(val_metrics["accuracy"])
        
        # Print epoch summary
        print(f"\nEpoch {epoch+1} Summary:")
        print(f"Train - Loss: {train_loss:.4f}, Acc: {train_metrics['accuracy']:.4f}, AUC: {train_metrics['roc_auc']:.4f}")
        print(f"Val   - Loss: {val_loss:.4f}, Acc: {val_metrics['accuracy']:.4f}, AUC: {val_metrics['roc_auc']:.4f}")
        
        # Check early stopping
        if val_metrics["roc_auc"] > best_val_auc:
            best_val_auc = val_metrics["roc_auc"]
            epochs_no_improve = 0
            torch.save({
                "epoch": epoch,
                "model_state_dict": cls_model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "best_val_auc": best_val_auc,
                "metrics": val_metrics
            }, os.path.join(args.output_dir, "checkpoints", "best_model.pth"))
            print(f"  --> Best model saved! Val AUC: {best_val_auc:.4f}")
        else:
            epochs_no_improve += 1
            print(f"  --> No improvement. Patience: {epochs_no_improve}/{args.early_stopping}")
            if epochs_no_improve >= args.early_stopping:
                print("  --> Early stopping triggered!")
                break
    
    # ========== Final Testing Phase ==========
    print("\nRunning final evaluation on test set...")
    # Load best model
    best_checkpoint = torch.load(
        os.path.join(args.output_dir, "checkpoints", "best_model.pth"), 
        map_location=device, weights_only=False
    )
    cls_model.load_state_dict(best_checkpoint["model_state_dict"])
    cls_model.eval()
    
    test_preds = []
    test_labels = []
    test_probs = []
    
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Testing"):
            images = batch["image"].to(device)
            labels = batch["label"].to(device)
            
            # Preprocess images
            processed_images = []
            for img in images:
                img_np = img.permute(1, 2, 0).cpu().numpy()
                mean = np.array([0.485, 0.456, 0.406])
                std = np.array([0.229, 0.224, 0.225])
                img_np = std * img_np + mean
                img_np = np.clip(img_np, 0, 1) * 255
                img_np = img_np.astype(np.uint8)
                
                if args.ablation == "seg_only":
                    processed = img_np
                elif args.ablation == "rgb":
                    pp = complete_preprocessing_pipeline(img_np, seg_model, device)
                    processed = pp["blended_rgb"]
                else:
                    pp = complete_preprocessing_pipeline(img_np, seg_model, device)
                    processed = pp["after_clahe"]
                
                from albumentations import Compose, Resize, Normalize
                from albumentations.pytorch import ToTensorV2
                transform = Compose([
                    Resize(args.image_size, args.image_size),
                    Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                    ToTensorV2()
                ])
                processed_tensor = transform(image=processed)["image"]
                processed_images.append(processed_tensor)
            
            processed_images = torch.stack(processed_images).to(device)
            logits, _ = cls_model(processed_images)
            probs = torch.softmax(logits, dim=1)
            preds = torch.argmax(probs, dim=1)
            
            test_preds.extend(preds.detach().cpu().numpy())
            test_labels.extend(labels.detach().cpu().numpy())
            test_probs.extend(probs[:, 1].detach().cpu().numpy())
    
    test_metrics = compute_classification_metrics(
        np.array(test_labels), np.array(test_preds), np.array(test_probs)
    )
    
    # Print test results
    print("\n" + "=" * 80)
    print("Test Set Results:")
    for k, v in test_metrics.items():
        if k != "confusion_matrix":
            print(f"  {k}: {v:.4f}")
    print("=" * 80)
    
    # Save plots
    plot_training_curves(
        train_losses, val_losses, train_accs, val_accs,
        save_path=os.path.join(args.output_dir, "figures", "training_curves.png")
    )
    plot_confusion_matrix(
        test_metrics["confusion_matrix"],
        save_path=os.path.join(args.output_dir, "figures", "confusion_matrix.png")
    )
    plot_roc_curve(
        np.array(test_labels), np.array(test_probs),
        save_path=os.path.join(args.output_dir, "figures", "roc_curve.png")
    )
    
    print(f"\nTraining complete! Results saved to {args.output_dir}")


if __name__ == "__main__":
    main()

