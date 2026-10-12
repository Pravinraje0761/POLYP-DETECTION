import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import os
from tqdm import tqdm

from src.models.a_denseunet import ADenseUNet, CombinedLoss
from src.data.dataloader import get_dataloaders
from src.utils.metrics import dice_score, iou_score, precision, recall, f1_score, AverageMeter


def train_epoch(model, loader, criterion, optimizer, device, epoch, writer):
    model.train()
    losses = AverageMeter()
    dices = AverageMeter()
    ious = AverageMeter()
    precisions = AverageMeter()
    recalls = AverageMeter()
    f1s = AverageMeter()
    
    pbar = tqdm(loader, desc=f"Epoch {epoch} Train")
    for images, masks in pbar:
        images = images.to(device)
        masks = masks.to(device)
        
        outputs = model(images)
        loss = criterion(outputs, masks)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        with torch.no_grad():
            d = dice_score(outputs, masks)
            i = iou_score(outputs, masks)
            p = precision(outputs, masks)
            r = recall(outputs, masks)
            f = f1_score(outputs, masks)
            
        losses.update(loss.item(), images.size(0))
        dices.update(d.item(), images.size(0))
        ious.update(i.item(), images.size(0))
        precisions.update(p.item(), images.size(0))
        recalls.update(r.item(), images.size(0))
        f1s.update(f.item(), images.size(0))
        
        pbar.set_postfix({"Loss": losses.avg, "Dice": dices.avg})
    
    writer.add_scalar("Train/Loss", losses.avg, epoch)
    writer.add_scalar("Train/Dice", dices.avg, epoch)
    writer.add_scalar("Train/IoU", ious.avg, epoch)
    writer.add_scalar("Train/Precision", precisions.avg, epoch)
    writer.add_scalar("Train/Recall", recalls.avg, epoch)
    writer.add_scalar("Train/F1", f1s.avg, epoch)
    
    return dices.avg


def val_epoch(model, loader, criterion, device, epoch, writer):
    model.eval()
    losses = AverageMeter()
    dices = AverageMeter()
    ious = AverageMeter()
    precisions = AverageMeter()
    recalls = AverageMeter()
    f1s = AverageMeter()
    
    with torch.no_grad():
        pbar = tqdm(loader, desc=f"Epoch {epoch} Val")
        for images, masks in pbar:
            images = images.to(device)
            masks = masks.to(device)
            
            outputs = model(images)
            loss = criterion(outputs, masks)
            
            d = dice_score(outputs, masks)
            i = iou_score(outputs, masks)
            p = precision(outputs, masks)
            r = recall(outputs, masks)
            f = f1_score(outputs, masks)
            
            losses.update(loss.item(), images.size(0))
            dices.update(d.item(), images.size(0))
            ious.update(i.item(), images.size(0))
            precisions.update(p.item(), images.size(0))
            recalls.update(r.item(), images.size(0))
            f1s.update(f.item(), images.size(0))
            
            pbar.set_postfix({"Loss": losses.avg, "Dice": dices.avg})
    
    writer.add_scalar("Val/Loss", losses.avg, epoch)
    writer.add_scalar("Val/Dice", dices.avg, epoch)
    writer.add_scalar("Val/IoU", ious.avg, epoch)
    writer.add_scalar("Val/Precision", precisions.avg, epoch)
    writer.add_scalar("Val/Recall", recalls.avg, epoch)
    writer.add_scalar("Val/F1", f1s.avg, epoch)
    
    return dices.avg


def main():
    # Configuration
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    batch_size = 8
    epochs = 100
    lr = 1e-4
    image_size = 224
    checkpoint_dir = "checkpoints"
    log_dir = "logs"
    
    os.makedirs(checkpoint_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)
    
    # TODO: Update these paths to your actual dataset
    train_img_dir = ""
    train_mask_dir = ""
    val_img_dir = ""
    val_mask_dir = ""
    
    # DataLoaders
    train_loader, val_loader = get_dataloaders(
        train_img_dir, train_mask_dir, val_img_dir, val_mask_dir,
        batch_size, image_size
    )
    
    # Model
    model = ADenseUNet(in_channels=3, out_channels=1).to(device)
    
    # Loss and optimizer
    criterion = CombinedLoss(bce_weight=0.5, dice_weight=0.5)
    optimizer = optim.AdamW(model.parameters(), lr=lr)
    
    # TensorBoard
    writer = SummaryWriter(log_dir=log_dir)
    
    # Training loop
    best_dice = 0.0
    
    for epoch in range(epochs):
        train_dice = train_epoch(model, train_loader, criterion, optimizer, device, epoch, writer)
        val_dice = val_epoch(model, val_loader, criterion, device, epoch, writer)
        
        if val_dice > best_dice:
            best_dice = val_dice
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "best_dice": best_dice,
            }, os.path.join(checkpoint_dir, "best_segmentation_model.pth"))
            print(f"Best model saved with Dice: {best_dice:.4f}")
    
    writer.close()


if __name__ == "__main__":
    main()
