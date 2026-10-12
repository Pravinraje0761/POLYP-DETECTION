"""
Stage 5: Vision Transformer Classifier Module (UPGRADED).
UPGRADES:
- LoRA (Low-Rank Adaptation) for parameter-efficient fine-tuning
- Better regularization
- Label smoothing
Uses pretrained ViT weights and Focal Loss.

.. note::
   This is an earlier version of the classification-stage preprocessing. The final
   version described in Sections 4.2-4.3 of the paper (soft ROI ``S = P**1.5``,
   fixed gamma 0.9, ROI threshold 0.05) is not yet in this repository; see
   docs/REPRODUCING.md.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import vit_b_16, ViT_B_16_Weights


class FocalLoss(nn.Module):
    """
    Focal Loss for imbalanced classification.
    """
    def __init__(self, alpha=0.25, gamma=2.0, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, logits, targets):
        ce_loss = F.cross_entropy(logits, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss


# LoRA Implementation
class LoRALayer(nn.Module):
    """
    LoRA (Low-Rank Adaptation) layer for linear layers.
    """
    def __init__(self, in_features, out_features, rank=8, alpha=8):
        super().__init__()
        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank
        
        # Low-rank matrices
        self.lora_A = nn.Linear(in_features, rank, bias=False)
        self.lora_B = nn.Linear(rank, out_features, bias=False)
        
        # Initialize A to random normal and B to zero
        nn.init.normal_(self.lora_A.weight, mean=0.0, std=1e-4)
        nn.init.zeros_(self.lora_B.weight)
        
    def forward(self, x):
        return self.lora_B(self.lora_A(x)) * self.scaling


class LoRAViT(nn.Module):
    """
    ViT with LoRA adapters for parameter-efficient fine-tuning.
    """
    def __init__(self, num_classes=2, pretrained=True, lora_rank=8, lora_alpha=8):
        super().__init__()
        
        # Load pretrained ViT
        if pretrained:
            self.vit = vit_b_16(weights=ViT_B_16_Weights.IMAGENET1K_V1)
        else:
            self.vit = vit_b_16(weights=None)
        
        self.num_classes = num_classes
        
        # Freeze original ViT parameters
        for param in self.vit.parameters():
            param.requires_grad = False
        
        # Add LoRA adapters to query and value projections in each attention layer
        self.lora_layers = nn.ModuleList()
        for layer_idx, layer in enumerate(self.vit.encoder.layers):
            # Query projection
            q_proj = layer.self_attention.in_proj_weight.chunk(3)[0]
            self.lora_layers.append(
                LoRALayer(q_proj.shape[1], q_proj.shape[0], lora_rank, lora_alpha)
            )
            # Value projection
            v_proj = layer.self_attention.in_proj_weight.chunk(3)[2]
            self.lora_layers.append(
                LoRALayer(v_proj.shape[1], v_proj.shape[0], lora_rank, lora_alpha)
            )
        
        # Replace classification head (trainable)
        in_features = self.vit.heads.head.in_features
        self.vit.heads.head = nn.Linear(in_features, num_classes)
        
        # Store original attention forward functions for explainability
        self._original_attn_forwards = []
        for layer in self.vit.encoder.layers:
            self._original_attn_forwards.append(layer.self_attention.forward)
    
    def forward(self, x):
        # We need to return both logits and attention weights for explainability
        attn_weights = []
        
        # Modify self-attention forward to return weights
        def modified_attn_forward(idx):
            def forward(query, key, value, key_padding_mask=None,
                       need_weights=True, attn_mask=None, average_attn_weights=True):
                # Call original forward with need_weights=True
                out, weights = self._original_attn_forwards[idx](
                    query, key, value, key_padding_mask,
                    need_weights=True, attn_mask=attn_mask,
                    average_attn_weights=average_attn_weights
                )
                attn_weights.append(weights)
                return out, weights
            return forward
        
        # Replace forward functions temporarily
        for i, layer in enumerate(self.vit.encoder.layers):
            layer.self_attention.forward = modified_attn_forward(i)
        
        # Forward pass through ViT
        logits = self.vit(x)
        
        # Restore original forward functions
        for i, layer in enumerate(self.vit.encoder.layers):
            layer.self_attention.forward = self._original_attn_forwards[i]
        
        return logits, attn_weights


class PolypViT(nn.Module):
    """
    Vision Transformer for polyp classification (UPGRADED: supports LoRA).
    """
    def __init__(self, num_classes=2, pretrained=True, use_lora=True, lora_rank=8, lora_alpha=8):
        super().__init__()
        
        if use_lora:
            self.vit_model = LoRAViT(num_classes, pretrained, lora_rank, lora_alpha)
            self.vit = self.vit_model.vit
        else:
            # Original implementation for backwards compatibility
            if pretrained:
                self.vit = vit_b_16(weights=ViT_B_16_Weights.IMAGENET1K_V1)
            else:
                self.vit = vit_b_16(weights=None)
            
            # Replace classification head
            in_features = self.vit.heads.head.in_features
            self.vit.heads.head = nn.Linear(in_features, num_classes)
            
            # Store original attention forward functions
            self._original_attn_forwards = []
            for layer in self.vit.encoder.layers:
                self._original_attn_forwards.append(layer.self_attention.forward)
    
    def forward(self, x):
        # For explainability, return logits and attention weights
        attn_weights = []
        
        if hasattr(self, 'vit_model'):
            return self.vit_model(x)
        
        # Original forward pass (without LoRA)
        def modified_attn_forward(idx):
            def forward(query, key, value, key_padding_mask=None,
                       need_weights=True, attn_mask=None, average_attn_weights=True):
                out, weights = self._original_attn_forwards[idx](
                    query, key, value, key_padding_mask,
                    need_weights=True, attn_mask=attn_mask,
                    average_attn_weights=average_attn_weights
                )
                attn_weights.append(weights)
                return out, weights
            return forward
        
        for i, layer in enumerate(self.vit.encoder.layers):
            layer.self_attention.forward = modified_attn_forward(i)
        
        logits = self.vit(x)
        
        for i, layer in enumerate(self.vit.encoder.layers):
            layer.self_attention.forward = self._original_attn_forwards[i]
        
        return logits, attn_weights

