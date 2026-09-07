import torch.nn as nn
import torch

class ChannelAttention(nn.Module):
    """Channel attention used in RCAN.
    Args:
        num_feat (int): Channel number of intermediate features.
        squeeze_factor (int): Channel squeeze factor. Default: 16.
    """

    def __init__(self, num_feat, squeeze_factor=16):
        super(ChannelAttention, self).__init__()
        self.attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(num_feat, num_feat // squeeze_factor, 1, padding=0),
            nn.ReLU(inplace=True),
            nn.Conv2d(num_feat // squeeze_factor, num_feat, 1, padding=0),
            nn.Sigmoid())

    def forward(self, x):
        y = self.attention(x)
        return x * y

class LoENm3D(nn.Module):
    def __init__(self, num_feat, compress_ratio=4,squeeze_factor=16):
        super(LoENm3D, self).__init__()
        self.cab = nn.Sequential(
            nn.Conv3d(num_feat, num_feat // compress_ratio, 3, 1, 1),
            nn.InstanceNorm3d(num_feat // compress_ratio), # 加入正则化
            nn.GELU(),
            nn.Conv3d(num_feat // compress_ratio, num_feat, 3, 1, 1),
            nn.InstanceNorm3d(num_feat), # 加入正则化
            ChannelAttention3D(num_feat, squeeze_factor)
        )

    def forward(self, x):
        return x + self.cab(x)
    
class LoE(nn.Module):
    def __init__(self, num_feat, compress_ratio=4,squeeze_factor=16):
        super(LoE, self).__init__()
        self.cab = nn.Sequential(
            nn.Conv2d(num_feat, num_feat // compress_ratio, 3, 1, 1),
            nn.GELU(),
            nn.Conv2d(num_feat // compress_ratio, num_feat, 3, 1, 1),
            ChannelAttention(num_feat, squeeze_factor)
        )

    def forward(self, x):
        return x + self.cab(x)

class ChannelAttention3D(nn.Module):
    """Channel attention used in RCAN.
        3D Channel Attention (RCAN-style)
        Input:  [B, C, D, H, W]
        Output: [B, C, D, H, W]
    """

    def __init__(self, num_feat, squeeze_factor=16):
        super(ChannelAttention3D, self).__init__()
        self.attention = nn.Sequential(
            nn.AdaptiveAvgPool3d(1),
            nn.Conv3d(num_feat, num_feat // squeeze_factor, 1, padding=0),
            nn.ReLU(inplace=True),
            nn.Conv3d(num_feat // squeeze_factor, num_feat, 1, padding=0),
            nn.Sigmoid())

    def forward(self, x):
        y = self.attention(x)
        return x * y

class LoE3D(nn.Module):
    def __init__(self, num_feat, compress_ratio=4,squeeze_factor=16):
        super(LoE3D, self).__init__()
        self.cab = nn.Sequential(
            nn.Conv3d(num_feat, num_feat // compress_ratio, 3, 1, 1),
            nn.GELU(),
            nn.Conv3d(num_feat // compress_ratio, num_feat, 3, 1, 1),
            ChannelAttention3D(num_feat, squeeze_factor)
        )

    def forward(self, x):
        return x + self.cab(x)



