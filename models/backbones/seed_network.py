import torch.nn as nn
from torch import Tensor

from configs.evolution_config import SeedNetworkConfig
from models.backbones. conv_block import ConvBlock


class SeedNetwork(nn.Module):
    
    def __init__(self, config: SeedNetworkConfig):
        super().__init__()
        
        self._config = config
        self._blocks = nn.ModuleList()
        self._channel_sizes:  list[int] = []
        
        self._build_architecture()
        
        self. global_pool = nn.AdaptiveAvgPool2d(1)
    
    def _build_architecture(self) -> None:
        in_channels = 3
        current_channels = self._config.initial_channels
        
        for _ in range(self._config.initial_blocks):
            block = ConvBlock(
                in_channels=in_channels,
                out_channels=current_channels,
                kernel_size=self._config.kernel_size,
                activation=self._config.activation,
                use_batch_norm=self._config.use_batch_norm,
                use_pooling=self._config.use_pooling,
            )
            
            self._blocks.append(block)
            self._channel_sizes.append(current_channels)
            
            in_channels = current_channels
            current_channels = min(current_channels * 2, 256)
    
    @property
    def feature_dim(self) -> int:
        return self._channel_sizes[-1] if self._channel_sizes else 0
    
    @property
    def num_blocks(self) -> int:
        return len(self._blocks)
    
    @property
    def channel_sizes(self) -> list[int]:
        return self._channel_sizes. copy()
    
    def forward(self, x: Tensor) -> Tensor:
        for block in self._blocks:
            x = block(x)
        
        x = self.global_pool(x)
        x = x.flatten(1)
        
        return x
    
    def get_architecture_summary(self) -> dict:
        total_params = sum(p.numel() for p in self. parameters())
        trainable_params = sum(
            p.numel() for p in self.parameters() if p.requires_grad
        )
        
        return {
            "num_blocks": self.num_blocks,
            "channel_progression": self._channel_sizes,
            "feature_dim": self.feature_dim,
            "total_params":  total_params,
            "trainable_params": trainable_params,
        }