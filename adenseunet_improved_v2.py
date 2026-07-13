
import torch
import torch.nn as nn
import torch.nn.functional as F


def get_norm(channels):
    groups = min(32, channels)
    while channels % groups != 0:
        groups -= 1
    return nn.GroupNorm(groups, channels)


class ASPPConv(nn.Sequential):
    def __init__(self, in_channels, out_channels, dilation):
        super().__init__(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=3,
                padding=dilation,
                dilation=dilation,
                bias=False
            ),
            get_norm(out_channels),
            nn.ReLU(inplace=True)
        )


class ASPPPooling(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Sequential(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=1,
                bias=False
            ),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        size = x.shape[-2:]
        x = self.pool(x)
        x = self.conv(x)
        x = F.interpolate(
            x,
            size=size,
            mode='bilinear',
            align_corners=False
        )
        return x


class ASPP(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels=256,
        rates=(1, 6, 12, 18)
    ):
        super().__init__()

        self.convs = nn.ModuleList()

        self.convs.append(
            nn.Sequential(
                nn.Conv2d(
                    in_channels,
                    out_channels,
                    kernel_size=1,
                    bias=False
                ),
                get_norm(out_channels),
                nn.ReLU(inplace=True)
            )
        )

        for rate in rates:
            self.convs.append(
                ASPPConv(
                    in_channels,
                    out_channels,
                    rate
                )
            )

        self.convs.append(
            ASPPPooling(
                in_channels,
                out_channels
            )
        )

        self.project = nn.Sequential(
            nn.Conv2d(
                (len(rates) + 2) * out_channels,
                out_channels,
                kernel_size=1,
                bias=False
            ),
            get_norm(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout2d(0.3)
        )

    def forward(self, x):
        res = [conv(x) for conv in self.convs]
        res = torch.cat(res, dim=1)
        return self.project(res)


class DenseBlock(nn.Module):
    def __init__(
        self,
        in_channels,
        growth_rate,
        num_layers,
        bn_size=4
    ):
        super().__init__()

        self.layers = nn.ModuleList()

        for _ in range(num_layers):
            self.layers.append(
                nn.Sequential(
                    get_norm(in_channels),
                    nn.ReLU(inplace=True),

                    nn.Conv2d(
                        in_channels,
                        bn_size * growth_rate,
                        kernel_size=1,
                        bias=False
                    ),

                    get_norm(
                        bn_size * growth_rate
                    ),
                    nn.ReLU(inplace=True),

                    nn.Conv2d(
                        bn_size * growth_rate,
                        growth_rate,
                        kernel_size=3,
                        padding=1,
                        bias=False
                    )
                )
            )

            in_channels += growth_rate

        self.out_channels = in_channels

    def forward(self, x):
        features = [x]

        for layer in self.layers:
            new_features = layer(
                torch.cat(features, dim=1)
            )
            features.append(new_features)

        return torch.cat(features, dim=1)


class AttentionBlock(nn.Module):
    def __init__(self, F_g, F_l, F_int):
        super().__init__()

        self.W_g = nn.Sequential(
            nn.Conv2d(
                F_g,
                F_int,
                kernel_size=1
            ),
            get_norm(F_int)
        )

        self.W_x = nn.Sequential(
            nn.Conv2d(
                F_l,
                F_int,
                kernel_size=1
            ),
            get_norm(F_int)
        )

        self.psi = nn.Sequential(
            nn.Conv2d(
                F_int,
                1,
                kernel_size=1
            ),
            nn.Sigmoid()
        )

        self.relu = nn.ReLU(inplace=True)

    def forward(self, g, x):
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        psi = self.relu(g1 + x1)
        psi = self.psi(psi)
        return x * psi


class DiceLoss(nn.Module):
    def __init__(self, smooth=1e-6):
        super().__init__()
        self.smooth = smooth

    def forward(self, logits, targets):
        probs = torch.sigmoid(logits)

        probs = probs.view(-1)
        targets = targets.view(-1)

        intersection = (probs * targets).sum()

        dice = (
            2.0 * intersection + self.smooth
        ) / (
            probs.sum() +
            targets.sum() +
            self.smooth
        )

        return 1.0 - dice


class FocalTverskyLoss(nn.Module):
    def __init__(
        self,
        alpha=0.7,
        beta=0.3,
        gamma=0.75,
        smooth=1e-6
    ):
        super().__init__()

        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.smooth = smooth

    def forward(self, logits, targets):
        probs = torch.sigmoid(logits)

        probs = probs.view(-1)
        targets = targets.view(-1)

        TP = (probs * targets).sum()
        FP = ((1 - targets) * probs).sum()
        FN = (targets * (1 - probs)).sum()

        tversky = (
            TP + self.smooth
        ) / (
            TP +
            self.alpha * FP +
            self.beta * FN +
            self.smooth
        )

        return (1 - tversky) ** self.gamma


class DeepSupervisionLoss(nn.Module):
    def __init__(
        self,
        final_weight=0.4,
        ds1_weight=0.2,
        ds2_weight=0.2,
        ds3_weight=0.2
    ):
        super().__init__()

        self.dice = DiceLoss()
        self.ftl = FocalTverskyLoss()

        self.final_weight = final_weight
        self.ds1_weight = ds1_weight
        self.ds2_weight = ds2_weight
        self.ds3_weight = ds3_weight

    def seg_loss(self, pred, target):
        return (
            0.5 * self.dice(pred, target)
            + 0.5 * self.ftl(pred, target)
        )

    def forward(
        self,
        final_out,
        ds1_out,
        ds2_out,
        ds3_out,
        targets
    ):
        loss_final = self.seg_loss(final_out, targets)
        loss_ds1 = self.seg_loss(ds1_out, targets)
        loss_ds2 = self.seg_loss(ds2_out, targets)
        loss_ds3 = self.seg_loss(ds3_out, targets)

        total_loss = (
            self.final_weight * loss_final
            + self.ds1_weight * loss_ds1
            + self.ds2_weight * loss_ds2
            + self.ds3_weight * loss_ds3
        )

        return total_loss


# --------------------------
# CBAM Attention Module
# --------------------------
class ChannelAttention(nn.Module):
    def __init__(self, in_channels, reduction_ratio=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        
        self.fc = nn.Sequential(
            nn.Linear(in_channels, in_channels // reduction_ratio, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(in_channels // reduction_ratio, in_channels, bias=False)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x).view(x.size(0), x.size(1)))
        max_out = self.fc(self.max_pool(x).view(x.size(0), x.size(1)))
        out = self.sigmoid(avg_out + max_out)
        return out.view(x.size(0), x.size(1), 1, 1)


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super().__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size=kernel_size, 
                             padding=kernel_size//2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv(x)
        return self.sigmoid(x)


class CBAM(nn.Module):
    def __init__(self, in_channels, reduction_ratio=16, kernel_size=7):
        super().__init__()
        self.ca = ChannelAttention(in_channels, reduction_ratio)
        self.sa = SpatialAttention(kernel_size)

    def forward(self, x):
        residual = x
        x = x * self.ca(x)
        x = x * self.sa(x)
        return x + residual  # Residual connection


# --------------------------
# Double Conv
# --------------------------
class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            get_norm(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            get_norm(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)


# --------------------------
# Improved A-DenseUNet
# --------------------------
class ADenseUNet_Improved(nn.Module):
    def __init__(self, in_channels=3, out_channels=1):
        super().__init__()
        
        # Encoder
        self.init_conv = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False),
            get_norm(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )
        
        # Encoder Block 1
        self.dense1 = DenseBlock(64, growth_rate=32, num_layers=4)
        self.cbam1 = CBAM(192)  # 64 + 4*32 = 192
        self.trans1 = nn.Sequential(
            get_norm(192),
            nn.ReLU(inplace=True),
            nn.Conv2d(192, 96, 1, bias=False),
            nn.MaxPool2d(2)
        )
        
        # Encoder Block 2
        self.dense2 = DenseBlock(96, growth_rate=32, num_layers=6)
        self.cbam2 = CBAM(288)  # 96 + 6*32 = 288
        self.trans2 = nn.Sequential(
            get_norm(288),
            nn.ReLU(inplace=True),
            nn.Conv2d(288, 144, 1, bias=False),
            nn.MaxPool2d(2)
        )
        
        # Encoder Block 3
        self.dense3 = DenseBlock(144, growth_rate=32, num_layers=8)
        self.cbam3 = CBAM(400)  # 144 + 8*32 = 400
        self.trans3 = nn.Sequential(
            get_norm(400),
            nn.ReLU(inplace=True),
            nn.Conv2d(400, 200, 1, bias=False),
            nn.MaxPool2d(2)
        )
        
        # ASPP Module
        self.aspp = ASPP(200, out_channels=256, rates=(1, 6, 12, 18))
        
        # Decoder 3
        self.up3 = nn.ConvTranspose2d(256, 200, kernel_size=2, stride=2)
        self.att3 = AttentionBlock(F_g=200, F_l=400, F_int=100)
        self.dec3 = DoubleConv(400 + 200, 200)
        
        # Decoder 2
        self.up2 = nn.ConvTranspose2d(200, 144, kernel_size=2, stride=2)
        self.att2 = AttentionBlock(F_g=144, F_l=288, F_int=72)
        self.dec2 = DoubleConv(288 + 144, 144)
        
        # Decoder 1
        self.up1 = nn.ConvTranspose2d(144, 96, kernel_size=2, stride=2)
        self.att1 = AttentionBlock(F_g=96, F_l=192, F_int=48)
        self.dec1 = DoubleConv(192 + 96, 96)
        
        # Final Decoder
        self.final_up = nn.ConvTranspose2d(96, 64, kernel_size=4, stride=4)
        self.final_conv = nn.Conv2d(64, out_channels, kernel_size=1)
        
        # Deep Supervision Heads
        self.ds3_conv = nn.Conv2d(200, out_channels, kernel_size=1)
        self.ds2_conv = nn.Conv2d(144, out_channels, kernel_size=1)
        self.ds1_conv = nn.Conv2d(96, out_channels, kernel_size=1)
        
    def forward(self, x, return_deep_supervision=False):
        # Encoder
        x1 = self.init_conv(x)  # (64, 56,56)
        
        x2 = self.dense1(x1)   # (192,56,56)
        x2 = self.cbam1(x2)    # CBAM + residual
        x2_t = self.trans1(x2) # (96,28,28)
        
        x3 = self.dense2(x2_t) # (288,28,28)
        x3 = self.cbam2(x3)    # CBAM + residual
        x3_t = self.trans2(x3) # (144,14,14)
        
        x4 = self.dense3(x3_t) # (400,14,14)
        x4 = self.cbam3(x4)    # CBAM + residual
        x4_t = self.trans3(x4) # (200,7,7)
        
        # ASPP
        aspp_out = self.aspp(x4_t) # (256,7,7)
        
        # Decoder 3
        d3 = self.up3(aspp_out)    # (200,14,14)
        x4_att = self.att3(d3, x4) # (400,14,14)
        d3 = torch.cat([d3, x4_att], dim=1) # (600,14,14)
        d3 = self.dec3(d3)         # (200,14,14)
        ds3_out = self.ds3_conv(d3)# Deep supervision 3
        
        # Decoder 2
        d2 = self.up2(d3)          # (144,28,28)
        x3_att = self.att2(d2, x3) # (288,28,28)
        d2 = torch.cat([d2, x3_att], dim=1) # (432,28,28)
        d2 = self.dec2(d2)         # (144,28,28)
        ds2_out = self.ds2_conv(d2)# Deep supervision 2
        
        # Decoder 1
        d1 = self.up1(d2)          # (96,56,56)
        x2_att = self.att1(d1, x2) # (192,56,56)
        d1 = torch.cat([d1, x2_att], dim=1) # (288,56,56)
        d1 = self.dec1(d1)         # (96,56,56)
        ds1_out = self.ds1_conv(d1)# Deep supervision 1
        
        # Final Decoder
        d0 = self.final_up(d1)     # (64,224,224)
        final_out = self.final_conv(d0)
        
        if return_deep_supervision:
            # Upsample deep supervision outputs to match final size
            ds3_out = F.interpolate(ds3_out, scale_factor=16, mode='bilinear', align_corners=False)
            ds2_out = F.interpolate(ds2_out, scale_factor=8, mode='bilinear', align_corners=False)
            ds1_out = F.interpolate(ds1_out, scale_factor=4, mode='bilinear', align_corners=False)
            return final_out, ds1_out, ds2_out, ds3_out
        else:
            return final_out
