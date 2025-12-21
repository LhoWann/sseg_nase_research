from typing import Literal

import torch. nn as nn
from torch import Tensor


class ConvBlock(nn.Module):
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        padding: int = 1,
        activation: Literal["relu", "gelu"] = "relu",
        use_batch_norm: bool = True,
        use_pooling: bool = True,
    ):
        super().__init__()
        
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            bias=not use_batch_norm,
        )
        
        self.bn = nn.BatchNorm2d(out_channels) if use_batch_norm else nn.Identity()
        
        if activation == "relu":
            self.activation = nn.ReLU(inplace=True)
        else:
            self.activation = nn.GELU()
        
        self.pool = nn.MaxPool2d(2, 2) if use_pooling else nn. Identity()
    
    def forward(self, x: Tensor) -> Tensor:
        x = self.conv(x)
        x = self.bn(x)
        x = self.activation(x)
        x = self.pool(x)
        return x


class ResidualConvBlock(nn. Module):
    
    def __init__(
        self,
        channels: int,
        kernel_size: int = 3,
        activation: Literal["relu", "gelu"] = "relu",
    ):
        super().__init__()
        
        padding = kernel_size // 2
        
        self.conv1 = nn.Conv2d(
            channels, channels, kernel_size, padding=padding, bias=False
        )
        self.bn1 = nn.BatchNorm2d(channels)
        
        self.conv2 = nn.Conv2d(
            channels, channels, kernel_size, padding=padding, bias=False
        )
        self.bn2 = nn.BatchNorm2d(channels)
        
        if activation == "relu":
            self.activation = nn.ReLU(inplace=True)
        else:
            self.activation = nn.GELU()
    
    def forward(self, x: Tensor) -> Tensor:
        identity = x
        
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.activation(out)
        
        out = self.conv2(out)
        out = self. bn2(out)
        
        out = out + identity
        out = self.activation(out)
        
        return out