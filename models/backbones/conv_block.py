from typing import Literal
import torch.nn as nn
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
        dropout: float = 0.3,
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
        self.pool = nn.MaxPool2d(2, 2) if use_pooling else nn.Identity()
        self.dropout = nn.Dropout2d(p=dropout) if dropout > 0 else nn.Identity()
    def forward(self, x: Tensor) -> Tensor:
        x = self.conv(x)
        x = self.bn(x)
        x = self.activation(x)
        x = self.dropout(x)
        if isinstance(self.pool, nn.MaxPool2d):
            if x.shape[-2] > 2 and x.shape[-1] > 2:
                x = self.pool(x)
        else:
            x = self.pool(x)
        return x
class ResidualConvBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        activation: Literal["relu", "gelu"] = "relu",
        dropout: float = 0.3,
        use_pooling: bool = False,
    ):
        super().__init__()
        padding = kernel_size // 2
        self.conv1 = nn.Conv2d(
            in_channels, out_channels, kernel_size, padding=padding, bias=False
        )
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(
            out_channels, out_channels, kernel_size, padding=padding, bias=False
        )
        self.bn2 = nn.BatchNorm2d(out_channels)
        if activation == "relu":
            self.activation = nn.ReLU(inplace=True)
        else:
            self.activation = nn.GELU()
        self.dropout = nn.Dropout2d(p=dropout) if dropout > 0 else nn.Identity()
        
        self.shortcut = nn.Identity()
        if in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x: Tensor) -> Tensor:
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.activation(out)
        out = self.dropout(out)
        out = self.conv2(out)
        out = self.bn2(out)
        if isinstance(self.shortcut, nn.Identity):
            out = out + identity
        else:
            out = out + self.shortcut(identity)
        out = self.activation(out)
        out = self.dropout(out)
        return out
