"""
A-DenseUNet: Adaptive Densely Connected UNet for Polyp Segmentation.
Fixed version with correct dimensions.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class DenseBlock(nn.Module):
    """
    Dense Block with dense connections.
    """
    def __init__(self, in_channels, growth_rate, num_layers, bn_size=4):
        super(DenseBlock, self).__init__()
        self.layers = nn.ModuleList()
        current_channels = in_channels
        for _ in range(num_layers):
            self.layers.append(self._make_layer(current_channels, growth_rate, bn_size))
            current_channels += growth_rate
        self.out_channels = current_channels

    def _make_layer(self, in_channels, growth_rate, bn_size):
        return nn.Sequential(
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, bn_size * growth_rate, kernel_size=1, bias=False),
            nn.BatchNorm2d(bn_size * growth_rate),
            nn.ReLU(inplace=True),
            nn.Conv2d(bn_size * growth_rate, growth_rate, kernel_size=3, padding=1, bias=False)
        )

    def forward(self, x):
        features = [x]
        for layer in self.layers:
            new_features = layer(torch.cat(features, 1))
            features.append(new_features)
        return torch.cat(features, 1)


class TransitionDown(nn.Module):
    """
    Transition Down block (bottleneck + pooling).
    """
    def __init__(self, in_channels, out_channels):
        super(TransitionDown, self).__init__()
        self.conv = nn.Sequential(
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

    def forward(self, x):
        return self.conv(x)


class AttentionBlock(nn.Module):
    """
    Attention Gate for decoder.
    """
    def __init__(self, F_g, F_l, F_int):
        super(AttentionBlock, self).__init__()
        self.W_g = nn.Sequential(
            nn.Conv2d(F_g, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )
        self.W_x = nn.Sequential(
            nn.Conv2d(F_l, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )
        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, kernel_size=1, stride=1, padding=0, bias=True),
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


class ResidualBlock(nn.Module):
    """
    Residual Block for decoder.
    """
    def __init__(self, in_channels, out_channels):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        if in_channels != out_channels:
            self.shortcut = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        else:
            self.shortcut = nn.Identity()

    def forward(self, x):
        residual = self.shortcut(x)
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out += residual
        out = self.relu(out)
        return out


class AtrousConvBlock(nn.Module):
    """
    Atrous Convolution Block for multi-scale feature extraction.
    """
    def __init__(self, in_channels, out_channels, dilation):
        super(AtrousConvBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, 
                              padding=dilation, dilation=dilation, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(self.bn(self.conv(x)))


class ADenseUNet(nn.Module):
    """
    Complete A-DenseUNet architecture (fixed).
    """
    def __init__(self, in_channels=3, out_channels=1, growth_rate=32, num_init_features=64):
        super(ADenseUNet, self).__init__()
        
        # Initial convolution
        self.init_conv = nn.Sequential(
            nn.Conv2d(in_channels, num_init_features, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(num_init_features),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )
        
        # Encoder
        self.dense1 = DenseBlock(num_init_features, growth_rate, num_layers=6)
        self.trans1 = TransitionDown(self.dense1.out_channels, 128)
        
        self.dense2 = DenseBlock(128, growth_rate, num_layers=12)
        self.trans2 = TransitionDown(self.dense2.out_channels, 256)
        
        self.dense3 = DenseBlock(256, growth_rate, num_layers=24)
        self.trans3 = TransitionDown(self.dense3.out_channels, 512)
        
        self.dense4 = DenseBlock(512, growth_rate, num_layers=16)
        
        # Atrous Convolutions
        self.atrous1 = AtrousConvBlock(self.dense4.out_channels, 512, dilation=2)
        self.atrous2 = AtrousConvBlock(512, 512, dilation=4)
        self.atrous3 = AtrousConvBlock(512, 512, dilation=6)
        
        # Decoder: use x4, x3, x2 (before transition down layers)
        self.up4 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.att4 = AttentionBlock(F_g=256, F_l=self.dense3.out_channels, F_int=128)
        self.res4 = ResidualBlock(self.dense3.out_channels + 256, 256)
        
        self.up3 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.att3 = AttentionBlock(F_g=128, F_l=self.dense2.out_channels, F_int=64)
        self.res3 = ResidualBlock(self.dense2.out_channels + 128, 128)
        
        self.up2 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.att2 = AttentionBlock(F_g=64, F_l=self.dense1.out_channels, F_int=32)
        self.res2 = ResidualBlock(self.dense1.out_channels + 64, 64)
        
        self.up1 = nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2)
        self.att1 = AttentionBlock(F_g=32, F_l=num_init_features, F_int=16)
        self.res1 = ResidualBlock(num_init_features + 32, 32)
        
        self.final_up = nn.ConvTranspose2d(32, 16, kernel_size=2, stride=2)
        self.final_conv = nn.Conv2d(16, out_channels, kernel_size=1)

    def forward(self, x):
        # Encoder
        x1 = self.init_conv(x)  # 64 channels, 56x56
        x2 = self.dense1(x1)    # dense1_out channels, 56x56
        x2_t = self.trans1(x2)  # 128 channels, 28x28
        x3 = self.dense2(x2_t)  # dense2_out channels, 28x28
        x3_t = self.trans2(x3)  # 256 channels, 14x14
        x4 = self.dense3(x3_t)  # dense3_out channels, 14x14
        x4_t = self.trans3(x4)  # 512 channels, 7x7
        x5 = self.dense4(x4_t)  # dense4_out channels, 7x7
        
        # Atrous
        x5 = self.atrous1(x5)
        x5 = self.atrous2(x5)
        x5 = self.atrous3(x5)
        
        # Decoder: use x4, x3, x2 (before transition down layers)
        d4 = self.up4(x5)  # upsample to 14x14
        x4_att = self.att4(d4, x4)
        d4 = torch.cat([d4, x4_att], dim=1)
        d4 = self.res4(d4)
        
        d3 = self.up3(d4)  # upsample to 28x28
        x3_att = self.att3(d3, x3)
        d3 = torch.cat([d3, x3_att], dim=1)
        d3 = self.res3(d3)
        
        d2 = self.up2(d3)  # upsample to 56x56
        x2_att = self.att2(d2, x2)
        d2 = torch.cat([d2, x2_att], dim=1)
        d2 = self.res2(d2)
        
        d1 = self.up1(d2)  # upsample to 112x112
        x1_att = self.att1(d1, x1)
        d1 = torch.cat([d1, x1_att], dim=1)
        d1 = self.res1(d1)
        
        d0 = self.final_up(d1)  # upsample to 224x224
        out = self.final_conv(d0)
        return out


class DiceLoss(nn.Module):
    """
    Dice Loss for segmentation.
    """
    def __init__(self, smooth=1e-6):
        super(DiceLoss, self).__init__()
        self.smooth = smooth

    def forward(self, logits, targets):
        logits = torch.sigmoid(logits)
        logits = logits.view(-1)
        targets = targets.view(-1)
        intersection = (logits * targets).sum()
        return 1 - (2. * intersection + self.smooth) / (logits.sum() + targets.sum() + self.smooth)


class CombinedLoss(nn.Module):
    """
    Combined Dice + BCE Loss.
    """
    def __init__(self, bce_weight=0.5, dice_weight=0.5):
        super(CombinedLoss, self).__init__()
        self.bce = nn.BCEWithLogitsLoss()
        self.dice = DiceLoss()
        self.bce_weight = bce_weight
        self.dice_weight = dice_weight

    def forward(self, logits, targets):
        bce_loss = self.bce(logits, targets)
        dice_loss = self.dice(logits, targets)
        return self.bce_weight * bce_loss + self.dice_weight * dice_loss
