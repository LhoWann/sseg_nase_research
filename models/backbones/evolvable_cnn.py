from typing import Optional

import torch
import torch.nn as nn
from torch import Tensor

from configs.evolution_config import EvolutionConfig
from configs.evolution_config import SeedNetworkConfig
from models.backbones.conv_block import ConvBlock
from models.backbones.seed_network import SeedNetwork


class EvolvableCNN(nn.Module):
    
    def __init__(
        self,
        seed_config: SeedNetworkConfig,
        evolution_config: EvolutionConfig,
        channel_progression: Optional[list[int]] = None,
    ):
        super().__init__()
        self._seed_config = seed_config
        self._evolution_config = evolution_config
        if channel_progression is not None:
            blocks = []
            self._channel_sizes = []
            in_channels = 3  # HARUS 3 untuk input RGB
            for out_channels in channel_progression:
                block = ConvBlock(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    kernel_size=self._seed_config.kernel_size,
                    activation=self._seed_config.activation,
                    use_batch_norm=self._seed_config.use_batch_norm,
                    use_pooling=self._seed_config.use_pooling,
                    dropout=getattr(self._seed_config, "dropout", 0.3),
                )
                blocks.append(block)
                self._channel_sizes.append(out_channels)
                in_channels = out_channels
            self._blocks = nn.ModuleList(blocks)
        else:
            seed = SeedNetwork(seed_config)
            self._blocks = seed._blocks
            self._channel_sizes = seed._channel_sizes
        self.global_pool = nn.AdaptiveAvgPool2d(1)
    
    def grow(self, out_channels: Optional[int] = None) -> bool:
        if len(self._blocks) >= self._evolution_config.growth.max_blocks:
            return False
        in_channels = self._channel_sizes[-1]
        if out_channels is None:
            out_channels = min(
                int(in_channels * self._evolution_config.growth.channel_expansion_ratio),
                self._evolution_config.growth.max_channels,
            )
        new_block = ConvBlock(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=self._seed_config.kernel_size,
            activation=self._seed_config.activation,
            use_batch_norm=self._seed_config.use_batch_norm,
            use_pooling=self._seed_config.use_pooling,
            dropout=getattr(self._seed_config, "dropout", 0.3),
        )

        ref_block = self._blocks[0] if len(self._blocks) > 0 else new_block
        device = next(ref_block.parameters()).device
        dtype = next(ref_block.parameters()).dtype
        new_block = new_block.to(device=device, dtype=dtype)
        self._initialize_identity(new_block, in_channels, out_channels)
        self._blocks.append(new_block)
        self._channel_sizes.append(out_channels)
        return True
    
    def _initialize_identity(
        self, block: ConvBlock, in_channels: int, out_channels: int
    ) -> None:
        nn.init.zeros_(block.conv.weight)
        
        min_channels = min(in_channels, out_channels)
        center = block.conv.kernel_size[0] // 2
        
        with torch.no_grad():
            for i in range(min_channels):
                block.conv.weight[i, i, center, center] = 1.0
    
    def widen(self, block_idx: int) -> bool:
        if block_idx >= len(self._blocks):
            return False
        old_channels = self._channel_sizes[block_idx]
        new_channels = min(
            int(old_channels * self._evolution_config.growth.channel_expansion_ratio),
            self._evolution_config.growth.max_channels,
        )
        if new_channels == old_channels: 
            return False
        in_channels = 3 if block_idx == 0 else self._channel_sizes[block_idx - 1]
        old_block = self._blocks[block_idx]

        device = next(old_block.parameters()).device
        dtype = next(old_block.parameters()).dtype
        new_block = ConvBlock(
            in_channels=in_channels,
            out_channels=new_channels,
            kernel_size=self._seed_config.kernel_size,
            activation=self._seed_config.activation,
            use_batch_norm=self._seed_config.use_batch_norm,
            use_pooling=self._seed_config.use_pooling,
            dropout=getattr(self._seed_config, "dropout", 0.3),
        ).to(device=device, dtype=dtype)
        with torch.no_grad():
            new_block.conv.weight[:old_channels] = old_block.conv.weight.to(dtype=dtype)
            if self._seed_config.use_batch_norm:
                new_block.bn.weight[:old_channels] = old_block.bn.weight.to(dtype=dtype)
                new_block.bn.bias[: old_channels] = old_block.bn.bias.to(dtype=dtype)
                new_block.bn.running_mean[:old_channels] = old_block.bn.running_mean.to(dtype=dtype)
                new_block.bn.running_var[:old_channels] = old_block.bn.running_var.to(dtype=dtype)
        self._blocks[block_idx] = new_block
        self._channel_sizes[block_idx] = new_channels
        if block_idx < len(self._blocks) - 1:
            self._update_next_block_input(block_idx, old_channels, new_channels)
        return True
    
    def _update_next_block_input(
        self, block_idx: int, old_in_channels: int, new_in_channels: int
    ) -> None:
        next_block = self._blocks[block_idx + 1]
        out_channels = self._channel_sizes[block_idx + 1]
        old_conv = next_block.conv
        device = old_conv.weight.device
        dtype = old_conv.weight.dtype
        new_conv = nn.Conv2d(
            new_in_channels,
            out_channels,
            kernel_size=old_conv.kernel_size,
            stride=old_conv.stride,
            padding=old_conv.padding,
            bias=old_conv.bias is not None,
        ).to(device=device, dtype=dtype)
        with torch.no_grad():
            new_conv.weight[:, :old_in_channels] = old_conv.weight.to(dtype=dtype)
            if old_conv.bias is not None:
                new_conv.bias.copy_(old_conv.bias.to(dtype=dtype))
        next_block.conv = new_conv
    
    @property
    def feature_dim(self) -> int:
        return self._channel_sizes[-1] if self._channel_sizes else 0
    
    @property
    def num_blocks(self) -> int:
        return len(self._blocks)
    
    @property
    def channel_sizes(self) -> list[int]:
        return self._channel_sizes.copy()
    
    def forward(self, x: Tensor) -> Tensor:
        for block in self._blocks:
            x = block(x)
        
        x = self.global_pool(x)
        x = x.flatten(1)
        
        return x
    
    def get_architecture_summary(self) -> dict:
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(
            p.numel() for p in self.parameters() if p.requires_grad
        )
        
        return {
            "num_blocks": self.num_blocks,
            "channel_progression": self._channel_sizes,
            "feature_dim": self.feature_dim,
            "total_params": total_params,
            "trainable_params":  trainable_params,
            "max_blocks": self._evolution_config.growth.max_blocks,
            "max_channels": self._evolution_config.growth.max_channels,
        }