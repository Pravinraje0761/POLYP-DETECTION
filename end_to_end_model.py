
"""
End-to-End Polyp Detection Model
Combines segmentation and classification
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import vit_b_16, ViT_B_16_Weights
from adenseunet_simple import ADenseUNet


class EndToEndPolypModel(nn.Module):
    def __init__(self, num_classes=2, pretrained_vit=True, freeze_segmentation=True):
        super().__init__()
        
        # Segmentation branch
        self.segmentation = ADenseUNet(in_channels=3, out_channels=1)
        
        # Load pretrained segmentation weights
        self._load_segmentation_weights()
        
        # Freeze segmentation if needed
        if freeze_segmentation:
            for param in self.segmentation.parameters():
                param.requires_grad = False
        
        # Image encoder (ViT)
        if pretrained_vit:
            self.vit = vit_b_16(weights=ViT_B_16_Weights.IMAGENET1K_V1)
        else:
            self.vit = vit_b_16(weights=None)
        
        # Remove the original classification head
        self.vit.heads = nn.Identity()
        
        # Mask encoder (processes segmentation mask)
        self.mask_encoder = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((7, 7)),
            nn.Flatten(),
            nn.Linear(128 * 7 * 7, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3)
        )
        
        # Combined classifier
        self.classifier = nn.Sequential(
            nn.Linear(768 + 512, 1024),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(1024, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(512, num_classes)
        )
    
    def _load_segmentation_weights(self):
        # Try to load improved model first
        try:
            checkpoint = torch.load("checkpoints/best_adenseunet_improved.pth", 
                                   map_location="cpu", weights_only=False)
            self.segmentation.load_state_dict(checkpoint["model_state_dict"])
            print("Successfully loaded improved segmentation weights")
            return
        except Exception as e:
            print(f"Improved model not found, trying original: {e}")
        
        # Fall back to original model
        try:
            checkpoint = torch.load("checkpoints/best_adenseunet.pth", 
                                   map_location="cpu", weights_only=False)
            self.segmentation.load_state_dict(checkpoint["model_state_dict"])
            print("Successfully loaded original segmentation weights")
        except Exception as e:
            print(f"Warning: Could not load any segmentation weights: {e}")
    
    def forward(self, x, return_mask=False):
        # Get segmentation mask
        with torch.set_grad_enabled(self.segmentation.training):
            seg_logits = self.segmentation(x)
            seg_mask = torch.sigmoid(seg_logits)
        
        # Encode mask
        mask_features = self.mask_encoder(seg_mask)
        
        # Encode image
        img_features = self.vit(x)
        
        # Combine features
        combined = torch.cat([img_features, mask_features], dim=1)
        
        # Classify
        logits = self.classifier(combined)
        
        if return_mask:
            return logits, seg_mask
        return logits


class FocalLoss(nn.Module):
    """
    Focal Loss for imbalanced classification
    """
    def __init__(self, alpha=0.25, gamma=2.0, reduction='mean'):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        
    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss
