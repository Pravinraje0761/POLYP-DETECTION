
"""
Ultimate Boundary-Aware A-DenseUNet with ASPP
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
        self.W_g = nn.Sequential(nn.Conv2d(F_g, F_int, 1), nn.BatchNorm2d(F_int))
        self.W_x = nn.Sequential(nn.Conv2d(F_l, F_int, 1), nn.BatchNorm2d(F_int))
        self.psi = nn.Sequential(nn.Conv2d(F_int, 1, 1), nn.BatchNorm2d(1), nn.Sigmoid())
        self.relu = nn.ReLU(inplace=True)
    def forward(self, g, x):
        g1, x1 = self.W_g(g), self.W_x(x)
        psi = self.relu(g1 + x1)
        psi = self.psi(psi)
        return x * psi

class DenseBlock(nn.Module):
    def __init__(self, in_channels, growth_rate, num_layers, bn_size=4):
        super().__init__()
        self.layers = nn.ModuleList()
        for _ in range(num_layers):
            self.layers.append(nn.Sequential(
                nn.BatchNorm2d(in_channels), nn.ReLU(inplace=True),
                nn.Conv2d(in_channels, bn_size * growth_rate, 1, bias=False),
                nn.BatchNorm2d(bn_size * growth_rate), nn.ReLU(inplace=True),
                nn.Conv2d(bn_size * growth_rate, growth_rate, 3, padding=1, bias=False)
            ))
            in_channels += growth_rate
        self.out_channels = in_channels
    def forward(self, x):
        features = [x]
        for layer in self.layers:
            new_features = layer(torch.cat(features, 1))
            features.append(new_features)
        return torch.cat(features, 1)

class ASPP(nn.Module):
    def __init__(self, in_channels, out_channels, rates=[6, 12, 18, 24]):
        super().__init__()
        self.branches = nn.ModuleList()
        self.branches.append(nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels), nn.ReLU(inplace=True)
        ))
        for rate in rates:
            self.branches.append(nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 3, padding=rate, dilation=rate, bias=False),
                nn.BatchNorm2d(out_channels), nn.ReLU(inplace=True)
            ))
        self.pool_branch = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels), nn.ReLU(inplace=True)
        )
        self.final = nn.Sequential(
            nn.Conv2d(out_channels * (len(rates) + 2), out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels), nn.ReLU(inplace=True)
        )
    def forward(self, x):
        size = x.shape[2:]
        outputs = []
        for branch in self.branches:
            outputs.append(branch(x))
        pool_out = F.interpolate(self.pool_branch(x), size=size, mode='bilinear', align_corners=False)
        outputs.append(pool_out)
        return self.final(torch.cat(outputs, 1))

class BoundaryRefinement(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, in_channels//2, 3, padding=1)
        self.conv2 = nn.Conv2d(in_channels//2, 1, 3, padding=1)
    def forward(self, x):
        x = F.relu(self.conv1(x))
        return torch.sigmoid(self.conv2(x))

class UltimateADenseUNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=1):
        super().__init__()
        # Encoder
        self.init_conv = nn.Sequential(
            nn.Conv2d(in_channels, 64, 7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(64), nn.ReLU(inplace=True),
            nn.MaxPool2d(3, stride=2, padding=1)
        )
        self.dense1 = DenseBlock(64, 48, 6)
        self.trans1 = nn.Sequential(nn.BatchNorm2d(64+6*48), nn.ReLU(), nn.Conv2d(64+6*48, (64+6*48)//2, 1, bias=False), nn.MaxPool2d(2))
        self.dense2 = DenseBlock((64+6*48)//2, 48, 8)
        self.trans2 = nn.Sequential(nn.BatchNorm2d(self.dense2.out_channels), nn.ReLU(), nn.Conv2d(self.dense2.out_channels, self.dense2.out_channels//2, 1, bias=False), nn.MaxPool2d(2))
        self.dense3 = DenseBlock(self.dense2.out_channels//2, 48, 12)
        self.trans3 = nn.Sequential(nn.BatchNorm2d(self.dense3.out_channels), nn.ReLU(), nn.Conv2d(self.dense3.out_channels, self.dense3.out_channels//2, 1, bias=False), nn.MaxPool2d(2))
        self.dense4 = DenseBlock(self.dense3.out_channels//2, 48, 16)
        self.trans4 = nn.Sequential(nn.BatchNorm2d(self.dense4.out_channels), nn.ReLU(), nn.Conv2d(self.dense4.out_channels, self.dense4.out_channels//2, 1, bias=False), nn.MaxPool2d(2))
        self.aspp = ASPP(self.dense4.out_channels//2, 512)
        
        # Decoder
        self.up4 = nn.ConvTranspose2d(512, self.dense4.out_channels//2, 2, stride=2)
        self.att4 = AttentionBlock(self.dense4.out_channels//2, self.dense4.out_channels, self.dense4.out_channels//2)
        self.dec4 = DoubleConv(self.dense4.out_channels + self.dense4.out_channels//2, self.dense4.out_channels//2)
        self.boundary4 = BoundaryRefinement(self.dense4.out_channels//2)
        
        self.up3 = nn.ConvTranspose2d(self.dense4.out_channels//2, self.dense3.out_channels//2, 2, stride=2)
        self.att3 = AttentionBlock(self.dense3.out_channels//2, self.dense3.out_channels, self.dense3.out_channels//2)
        self.dec3 = DoubleConv(self.dense3.out_channels + self.dense3.out_channels//2, self.dense3.out_channels//2)
        self.boundary3 = BoundaryRefinement(self.dense3.out_channels//2)
        
        self.up2 = nn.ConvTranspose2d(self.dense3.out_channels//2, self.dense2.out_channels//2, 2, stride=2)
        self.att2 = AttentionBlock(self.dense2.out_channels//2, self.dense2.out_channels, self.dense2.out_channels//2)
        self.dec2 = DoubleConv(self.dense2.out_channels + self.dense2.out_channels//2, self.dense2.out_channels//2)
        self.boundary2 = BoundaryRefinement(self.dense2.out_channels//2)
        
        self.up1 = nn.ConvTranspose2d(self.dense2.out_channels//2, (64+6*48)//2, 2, stride=2)
        self.att1 = AttentionBlock((64+6*48)//2, 64+6*48, (64+6*48)//4)
        self.dec1 = DoubleConv(64+6*48 + (64+6*48)//2, (64+6*48)//2)
        
        self.final_up = nn.ConvTranspose2d((64+6*48)//2, 64, 4, stride=4)
        self.final_conv = nn.Conv2d(64, out_channels, 1)
        
    def forward(self, x):
        x1 = self.init_conv(x)
        x2 = self.dense1(x1)
        x2_t = self.trans1(x2)
        x3 = self.dense2(x2_t)
        x3_t = self.trans2(x3)
        x4 = self.dense3(x3_t)
        x4_t = self.trans3(x4)
        x5 = self.dense4(x4_t)
        x5_t = self.trans4(x5)
        bn = self.aspp(x5_t)
        
        d4 = self.up4(bn)
        x5_att = self.att4(d4, x5)
        d4 = torch.cat([d4, x5_att], dim=1)
        d4 = self.dec4(d4)
        b4 = self.boundary4(d4)
        
        d3 = self.up3(d4)
        x4_att = self.att3(d3, x4)
        d3 = torch.cat([d3, x4_att], dim=1)
        d3 = self.dec3(d3)
        b3 = self.boundary3(d3)
        
        d2 = self.up2(d3)
        x3_att = self.att2(d2, x3)
        d2 = torch.cat([d2, x3_att], dim=1)
        d2 = self.dec2(d2)
        b2 = self.boundary2(d2)
        
        d1 = self.up1(d2)
        x2_att = self.att1(d1, x2)
        d1 = torch.cat([d1, x2_att], dim=1)
        d1 = self.dec1(d1)
        
        d0 = self.final_up(d1)
        out = self.final_conv(d0)
        
        return out, [b4, b3, b2]

