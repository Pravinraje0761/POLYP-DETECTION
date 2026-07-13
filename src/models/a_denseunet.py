import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models


class DenseBlock(nn.Module):
    def __init__(self, in_channels, growth_rate, num_layers):
        super(DenseBlock, self).__init__()
        self.layers = nn.ModuleList()
        current_channels = in_channels
        for _ in range(num_layers):
            self.layers.append(self._make_layer(current_channels, growth_rate))
            current_channels += growth_rate
        self.out_channels = current_channels

    def _make_layer(self, in_channels, growth_rate):
        return nn.Sequential(
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, 4 * growth_rate, kernel_size=1, bias=False),
            nn.BatchNorm2d(4 * growth_rate),
            nn.ReLU(inplace=True),
            nn.Conv2d(4 * growth_rate, growth_rate, kernel_size=3, padding=1, bias=False)
        )

    def forward(self, x):
        features = [x]
        for layer in self.layers:
            new_features = layer(torch.cat(features, 1))
            features.append(new_features)
        return torch.cat(features, 1)


class TransitionDown(nn.Module):
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
    def __init__(self, in_channels, out_channels, dilation):
        super(AtrousConvBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, 
                              padding=dilation, dilation=dilation, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(self.bn(self.conv(x)))


class ADenseUNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=1, growth_rate=32):
        super(ADenseUNet, self).__init__()
        
        # Encoder
        self.init_conv = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )
        
        self.dense1 = DenseBlock(64, growth_rate, 6)
        self.trans1 = TransitionDown(self.dense1.out_channels, 256)
        
        self.dense2 = DenseBlock(256, growth_rate, 12)
        self.trans2 = TransitionDown(self.dense2.out_channels, 512)
        
        self.dense3 = DenseBlock(512, growth_rate, 24)
        self.trans3 = TransitionDown(self.dense3.out_channels, 1024)
        
        self.dense4 = DenseBlock(1024, growth_rate, 16)
        
        # Atrous convolutions
        self.atrous1 = AtrousConvBlock(self.dense4.out_channels, 512, 2)
        self.atrous2 = AtrousConvBlock(512, 512, 4)
        self.atrous3 = AtrousConvBlock(512, 512, 6)
        
        # Decoder
        self.up4 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.att4 = AttentionBlock(256, 1024, 128)
        self.res4 = ResidualBlock(1024 + 256, 256)
        
        self.up3 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.att3 = AttentionBlock(128, 512, 64)
        self.res3 = ResidualBlock(512 + 128, 128)
        
        self.up2 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.att2 = AttentionBlock(64, 256, 32)
        self.res2 = ResidualBlock(256 + 64, 64)
        
        self.up1 = nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2)
        self.att1 = AttentionBlock(32, 64, 16)
        self.res1 = ResidualBlock(64 + 32, 32)
        
        self.final_up = nn.ConvTranspose2d(32, 16, kernel_size=2, stride=2)
        self.final_conv = nn.Conv2d(16, out_channels, kernel_size=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # Encoder
        x1 = self.init_conv(x)
        x2 = self.dense1(x1)
        x2_t = self.trans1(x2)
        x3 = self.dense2(x2_t)
        x3_t = self.trans2(x3)
        x4 = self.dense3(x3_t)
        x4_t = self.trans3(x4)
        x5 = self.dense4(x4_t)
        
        # Atrous
        x5 = self.atrous1(x5)
        x5 = self.atrous2(x5)
        x5 = self.atrous3(x5)
        
        # Decoder
        d4 = self.up4(x5)
        x4_att = self.att4(d4, x4)
        d4 = torch.cat([d4, x4_att], dim=1)
        d4 = self.res4(d4)
        
        d3 = self.up3(d4)
        x3_att = self.att3(d3, x3)
        d3 = torch.cat([d3, x3_att], dim=1)
        d3 = self.res3(d3)
        
        d2 = self.up2(d3)
        x2_att = self.att2(d2, x2)
        d2 = torch.cat([d2, x2_att], dim=1)
        d2 = self.res2(d2)
        
        d1 = self.up1(d2)
        x1_att = self.att1(d1, x1)
        d1 = torch.cat([d1, x1_att], dim=1)
        d1 = self.res1(d1)
        
        d0 = self.final_up(d1)
        out = self.final_conv(d0)
        out = self.sigmoid(out)
        return out


class DiceLoss(nn.Module):
    def __init__(self, smooth=1e-6):
        super(DiceLoss, self).__init__()
        self.smooth = smooth

    def forward(self, logits, targets):
        logits = logits.view(-1)
        targets = targets.view(-1)
        intersection = (logits * targets).sum()
        return 1 - (2. * intersection + self.smooth) / (logits.sum() + targets.sum() + self.smooth)


class CombinedLoss(nn.Module):
    def __init__(self, bce_weight=0.5, dice_weight=0.5):
        super(CombinedLoss, self).__init__()
        self.bce = nn.BCELoss()
        self.dice = DiceLoss()
        self.bce_weight = bce_weight
        self.dice_weight = dice_weight

    def forward(self, logits, targets):
        bce_loss = self.bce(logits, targets)
        dice_loss = self.dice(logits, targets)
        return self.bce_weight * bce_loss + self.dice_weight * dice_loss
