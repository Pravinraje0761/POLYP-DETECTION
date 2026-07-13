"""
Simple A-DenseUNet for Polyp Segmentation (correct dimensions).
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)


class AttentionBlock(nn.Module):
    def __init__(self, F_g, F_l, F_int):
        super().__init__()
        self.W_g = nn.Sequential(
            nn.Conv2d(F_g, F_int, kernel_size=1),
            nn.BatchNorm2d(F_int)
        )
        self.W_x = nn.Sequential(
            nn.Conv2d(F_l, F_int, kernel_size=1),
            nn.BatchNorm2d(F_int)
        )
        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, kernel_size=1),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, g, x):
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        psi = self.relu(g1 + x1)
        psi = self.psi(psi)
        return x * psi


class DenseBlock(nn.Module):
    def __init__(self, in_channels, growth_rate, num_layers, bn_size=4):
        super().__init__()
        self.layers = nn.ModuleList()
        for _ in range(num_layers):
            self.layers.append(nn.Sequential(
                nn.BatchNorm2d(in_channels),
                nn.ReLU(inplace=True),
                nn.Conv2d(in_channels, bn_size * growth_rate, kernel_size=1, bias=False),
                nn.BatchNorm2d(bn_size * growth_rate),
                nn.ReLU(inplace=True),
                nn.Conv2d(bn_size * growth_rate, growth_rate, kernel_size=3, padding=1, bias=False)
            ))
            in_channels += growth_rate
        self.out_channels = in_channels

    def forward(self, x):
        features = [x]
        for layer in self.layers:
            new_features = layer(torch.cat(features, 1))
            features.append(new_features)
        return torch.cat(features, 1)


class ADenseUNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=1):
        super().__init__()
        
        # Encoder
        self.init_conv = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )
        
        self.dense1 = DenseBlock(64, growth_rate=32, num_layers=4)  # Output: 64+4*32=192
        self.trans1 = nn.Sequential(
            nn.BatchNorm2d(192),
            nn.ReLU(inplace=True),
            nn.Conv2d(192, 96, 1, bias=False),
            nn.MaxPool2d(2)
        )
        
        self.dense2 = DenseBlock(96, growth_rate=32, num_layers=6)  # Output: 96+6*32=288
        self.trans2 = nn.Sequential(
            nn.BatchNorm2d(288),
            nn.ReLU(inplace=True),
            nn.Conv2d(288, 144, 1, bias=False),
            nn.MaxPool2d(2)
        )
        
        self.dense3 = DenseBlock(144, growth_rate=32, num_layers=8)  # Output:144+8*32=400
        self.trans3 = nn.Sequential(
            nn.BatchNorm2d(400),
            nn.ReLU(inplace=True),
            nn.Conv2d(400, 200, 1, bias=False),
            nn.MaxPool2d(2)
        )
        
        self.bottleneck = DoubleConv(200, 400)
        
        # Decoder
        self.up3 = nn.ConvTranspose2d(400, 200, kernel_size=2, stride=2)
        self.att3 = AttentionBlock(F_g=200, F_l=400, F_int=100)
        self.dec3 = DoubleConv(400 + 200, 200)
        
        self.up2 = nn.ConvTranspose2d(200, 144, kernel_size=2, stride=2)
        self.att2 = AttentionBlock(F_g=144, F_l=288, F_int=72)
        self.dec2 = DoubleConv(288 + 144, 144)
        
        self.up1 = nn.ConvTranspose2d(144, 96, kernel_size=2, stride=2)
        self.att1 = AttentionBlock(F_g=96, F_l=192, F_int=48)
        self.dec1 = DoubleConv(192 + 96, 96)
        
        self.final_up = nn.ConvTranspose2d(96, 64, kernel_size=4, stride=4)
        self.final_conv = nn.Conv2d(64, out_channels, kernel_size=1)
        
    def forward(self, x):
        # Encoder
        x1 = self.init_conv(x)  # (64, 56,56)
        x2 = self.dense1(x1)   # (192,56,56)
        x2_t = self.trans1(x2) # (96,28,28)
        x3 = self.dense2(x2_t) # (288,28,28)
        x3_t = self.trans2(x3) # (144,14,14)
        x4 = self.dense3(x3_t) # (400,14,14)
        x4_t = self.trans3(x4) # (200,7,7)
        
        # Bottleneck
        bn = self.bottleneck(x4_t) # (400,7,7)
        
        # Decoder
        d3 = self.up3(bn)          # (200,14,14)
        x4_att = self.att3(d3, x4) # (400,14,14)
        d3 = torch.cat([d3, x4_att], dim=1) # (600,14,14)
        d3 = self.dec3(d3)         # (200,14,14)
        
        d2 = self.up2(d3)          # (144,28,28)
        x3_att = self.att2(d2, x3) # (288,28,28)
        d2 = torch.cat([d2, x3_att], dim=1) # (432,28,28)
        d2 = self.dec2(d2)         # (144,28,28)
        
        d1 = self.up1(d2)          # (96,56,56)
        x2_att = self.att1(d1, x2) # (192,56,56)
        d1 = torch.cat([d1, x2_att], dim=1) # (288,56,56)
        d1 = self.dec1(d1)         # (96,56,56)
        
        d0 = self.final_up(d1)     # (64,224,224)
        out = self.final_conv(d0)
        return out


class DiceLoss(nn.Module):
    def __init__(self, smooth=1e-6):
        super().__init__()
        self.smooth = smooth

    def forward(self, logits, targets):
        # logits are probabilities (from sigmoid)
        logits = logits.view(-1)
        targets = targets.view(-1)
        intersection = (logits * targets).sum()
        return 1 - (2. * intersection + self.smooth) / (logits.sum() + targets.sum() + self.smooth)


class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, probs, targets):
        probs = probs.view(-1)
        targets = targets.view(-1)
        bce = F.binary_cross_entropy(probs, targets, reduction='none')
        pt = torch.where(targets == 1, probs, 1 - probs)
        focal = self.alpha * (1 - pt) ** self.gamma * bce
        return focal.mean()


class CombinedLoss(nn.Module):
    def __init__(self, bce_weight=0.2, dice_weight=0.6, focal_weight=0.2):
        super().__init__()
        self.bce = nn.BCELoss()
        self.dice = DiceLoss()
        self.focal = FocalLoss()
        self.bce_weight = bce_weight
        self.dice_weight = dice_weight
        self.focal_weight = focal_weight

    def forward(self, logits, targets):
        probs = torch.sigmoid(logits)
        bce_loss = self.bce(probs, targets)
        dice_loss = self.dice(probs, targets)
        focal_loss = self.focal(probs, targets)
        return (
            self.bce_weight * bce_loss +
            self.dice_weight * dice_loss +
            self.focal_weight * focal_loss
        )
