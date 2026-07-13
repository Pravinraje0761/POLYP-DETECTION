
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from scipy.ndimage import distance_transform_edt

class FocalTverskyLoss(nn.Module):
    def __init__(self, alpha=0.3, beta=0.7, gamma=0.75, smooth=1e-6):
        super(FocalTverskyLoss, self).__init__()
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.smooth = smooth

    def forward(self, logits, targets):
        """
        Args:
            logits: (B, 1, H, W) - raw logits from model
            targets: (B, 1, H, W) - binary ground truth
        """
        probs = torch.sigmoid(logits)
        
        # Flatten
        probs_flat = probs.view(-1)
        targets_flat = targets.view(-1)
        
        # True positives, false positives, false negatives
        TP = (probs_flat * targets_flat).sum()
        FP = ((1 - targets_flat) * probs_flat).sum()
        FN = (targets_flat * (1 - probs_flat)).sum()
        
        # Tversky index
        tversky = (TP + self.smooth) / (TP + self.alpha * FP + self.beta * FN + self.smooth)
        
        # Focal Tversky loss
        loss = (1 - tversky) ** self.gamma
        return loss


class BoundaryLoss(nn.Module):
    def __init__(self, smooth=1e-6):
        super(BoundaryLoss, self).__init__()
        self.smooth = smooth

    def forward(self, logits, targets):
        probs = torch.sigmoid(logits)
        batch_size = probs.shape[0]
        
        total_loss = 0.0
        
        for i in range(batch_size):
            target_np = targets[i, 0].detach().cpu().numpy()
            prob_np = probs[i, 0].detach().cpu().numpy()
            
            # Compute distance maps
            pos_dist_map = distance_transform_edt(1 - target_np)
            neg_dist_map = distance_transform_edt(target_np)
            
            # Boundary distance map
            boundary_dist_map = target_np * pos_dist_map + (1 - target_np) * neg_dist_map
            boundary_dist_map = torch.tensor(boundary_dist_map, dtype=torch.float32, device=probs.device)
            
            # Compute loss
            loss = (boundary_dist_map * prob_np).sum()
            total_loss += loss
        
        return total_loss / batch_size


class CombinedLoss(nn.Module):
    def __init__(
        self, 
        ft_alpha=0.3, 
        ft_beta=0.7, 
        ft_gamma=0.75, 
        lambda_ft=0.7, 
        lambda_boundary=0.3
    ):
        super(CombinedLoss, self).__init__()
        self.focal_tversky = FocalTverskyLoss(alpha=ft_alpha, beta=ft_beta, gamma=ft_gamma)
        self.boundary_loss = BoundaryLoss()
        self.lambda_ft = lambda_ft
        self.lambda_boundary = lambda_boundary

    def forward(self, logits, targets):
        ft_loss = self.focal_tversky(logits, targets)
        bd_loss = self.boundary_loss(logits, targets)
        return self.lambda_ft * ft_loss + self.lambda_boundary * bd_loss
