
"""
A-DenseUNet: the lightweight segmentation model evaluated in the paper (Section 4.1).

U-Net-shaped encoder-decoder with a 7x7 stride-2 stem (64 channels), three densely
connected encoder blocks (growth rate 32; 4, 6 and 8 layers), a DoubleConv bottleneck,
and three decoder stages whose skip connections pass through additive attention gates.
Output is a single-channel logit map at the input resolution.

Parameter count: 6,584,870 (verified in tests/test_models.py), which is the figure
reported in Table 15. Trained with ``polypgate.losses.CombinedLoss``
(0.7 x Focal Tversky + 0.3 x boundary loss) at 320x320.

The class keeps its original name ``ADenseUNet_Boosted`` so existing checkpoints load
unchanged; ``ADenseUNet`` is provided as an alias.
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


class ADenseUNet_Boosted(nn.Module):
    def __init__(self, in_channels=3, out_channels=1):
        super().__init__()
        
        # Encoder
        self.init_conv = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )
        
        self.dense1 = DenseBlock(64, growth_rate=32, num_layers=4)
        self.trans1 = nn.Sequential(
            nn.BatchNorm2d(192),
            nn.ReLU(inplace=True),
            nn.Conv2d(192, 96, 1, bias=False),
            nn.MaxPool2d(2)
        )
        
        self.dense2 = DenseBlock(96, growth_rate=32, num_layers=6)
        self.trans2 = nn.Sequential(
            nn.BatchNorm2d(288),
            nn.ReLU(inplace=True),
            nn.Conv2d(288, 144, 1, bias=False),
            nn.MaxPool2d(2)
        )
        
        self.dense3 = DenseBlock(144, growth_rate=32, num_layers=8)
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
        
        # Final upsampling (keys match best_adenseunet_boosted*.pth checkpoints)
        self.up_final = nn.ConvTranspose2d(96, 64, kernel_size=2, stride=2)
        self.final_conv = nn.Conv2d(64, out_channels, kernel_size=1)
        
    def forward(self, x):
        # Encoder
        x1 = self.init_conv(x)
        x2 = self.dense1(x1)
        x2_t = self.trans1(x2)
        x3 = self.dense2(x2_t)
        x3_t = self.trans2(x3)
        x4 = self.dense3(x3_t)
        x4_t = self.trans3(x4)
        
        # Bottleneck
        bn = self.bottleneck(x4_t)
        
        # Decoder
        d3 = self.up3(bn)
        x4_att = self.att3(d3, x4)
        d3 = torch.cat([d3, x4_att], dim=1)
        d3 = self.dec3(d3)
        
        d2 = self.up2(d3)
        x3_att = self.att2(d2, x3)
        d2 = torch.cat([d2, x3_att], dim=1)
        d2 = self.dec2(d2)
        
        d1 = self.up1(d2)
        x2_att = self.att1(d1, x2)
        d1 = torch.cat([d1, x2_att], dim=1)
        d1 = self.dec1(d1)
        
        d0 = self.up_final(d1)
        out = self.final_conv(d0)
        if out.shape[-2:] != x.shape[-2:]:
            out = F.interpolate(out, size=x.shape[-2:], mode="bilinear", align_corners=False)
        return out


if __name__ == "__main__":
    # Quick test to make sure it's compatible with the checkpoint
    model = ADenseUNet_Boosted(in_channels=3, out_channels=1)
    checkpoint = torch.load("checkpoints/best_adenseunet_boosted.pth", map_location="cpu", weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    print("✅ Model loaded successfully!")
    print(f"   Best Dice Score: {checkpoint['best_dice']:.4f}")


# Public alias used in the paper and the README.
ADenseUNet = ADenseUNet_Boosted
