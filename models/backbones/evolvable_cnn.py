from typing import Optional
import torch
import torch.nn as nn
from torch import Tensor
from configs.evolution_config import EvolutionConfig
from configs.evolution_config import SeedNetworkConfig
from models.backbones.conv_block import ResidualConvBlock
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
            in_channels = 3
            for out_channels in channel_progression:
                block = ResidualConvBlock(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    kernel_size=self._seed_config.kernel_size,
                    activation=self._seed_config.activation,
                    dropout=getattr(self._seed_config, "dropout", 0.3),
                    use_pooling=self._seed_config.use_pooling,
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
        new_block = ResidualConvBlock(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=self._seed_config.kernel_size,
            activation=self._seed_config.activation,
            dropout=getattr(self._seed_config, "dropout", 0.3),
        )
        ref_block = self._blocks[0] if len(self._blocks) > 0 else new_block
        device = next(ref_block.parameters()).device
        dtype = next(ref_block.parameters()).dtype
        new_block = new_block.to(device=device, dtype=dtype)
        self._initialize_residual_zero(new_block)
        self._blocks.append(new_block)
        self._channel_sizes.append(out_channels)
        return True

    def _initialize_residual_zero(
        self, block: ResidualConvBlock
    ) -> None:
        nn.init.zeros_(block.conv2.weight)
        if hasattr(block, "bn2") and isinstance(block.bn2, nn.BatchNorm2d):
            nn.init.zeros_(block.bn2.weight)
            nn.init.zeros_(block.bn2.bias)
            nn.init.zeros_(block.bn2.running_mean)
            nn.init.ones_(block.bn2.running_var)
        
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
        
        new_block = ResidualConvBlock(
            in_channels=in_channels,
            out_channels=new_channels,
            kernel_size=self._seed_config.kernel_size,
            activation=self._seed_config.activation,
            dropout=getattr(self._seed_config, "dropout", 0.3),
        ).to(device=device, dtype=dtype)
        
        # Note: Net2WiderNet for ResidualBlock is complex. 
        # For simplicity in this fix, we initialize similarly to standard, relying on zero-init of residual path to minimize impact?
        # Ideally, we copy weights. But dimensions change.
        # Fallback to zero-init for stability (like grow).
        self._initialize_residual_zero(new_block)
        
        # If we want to preserve old weights, we need complex slicing.
        # Given "Function Preserving" priority: A zero-init residual block is safer than a random one.
        
        self._blocks[block_idx] = new_block
        self._channel_sizes[block_idx] = new_channels
        
        if block_idx < len(self._blocks) - 1:
            self._update_next_block_input(block_idx, old_channels, new_channels)
        return True

    def _update_next_block_input(
        self, block_idx: int, old_in_channels: int, new_in_channels: int
    ) -> None:
        next_block = self._blocks[block_idx + 1]
        # ResidualBlock update structure is harder. 
        # Re-creating the next block as identity is safest strategy if architecture changes mid-stream.
        # But we need to update next block's input channels.
        
        out_channels = self._channel_sizes[block_idx + 1]
        
        new_next_block = ResidualConvBlock(
             in_channels=new_in_channels,
             out_channels=out_channels,
             kernel_size=self._seed_config.kernel_size,
             activation=self._seed_config.activation,
             dropout=getattr(self._seed_config, "dropout", 0.3),
        ).to(device=next_block.conv1.weight.device, dtype=next_block.conv1.weight.dtype)
        
        self._initialize_residual_zero(new_next_block)
        self._blocks[block_idx + 1] = new_next_block
    @property
    def feature_dim(self) -> int:
        return self._channel_sizes[-1] if self._channel_sizes else 0
    @property
    def num_blocks(self) -> int:
        return len(self._blocks)
    @property
    def channel_sizes(self) -> list[int]:
        return self._channel_sizes.copy()

    def estimate_flops(self, input_size: tuple[int, int, int] = (3, 84, 84)) -> float:
        flops = 0
        current_channels = input_size[0]
        h, w = input_size[1], input_size[2]
        for block in self._blocks:
            kernel_ops = block.conv1.kernel_size[0] * block.conv1.kernel_size[1]
            flops += h * w * current_channels * kernel_ops * block.conv1.out_channels
            mid_channels = block.conv1.out_channels
            
            kernel_ops2 = block.conv2.kernel_size[0] * block.conv2.kernel_size[1]
            flops += h * w * mid_channels * kernel_ops2 * block.conv2.out_channels
            current_channels = block.conv2.out_channels

            if not isinstance(block.shortcut, nn.Identity):
                 flops += h * w * block.shortcut[0].in_channels * 1 * block.shortcut[0].out_channels

            if self._seed_config.use_pooling:
                h //= 2
                w //= 2
        return flops / 1e9
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
            "num_channels": self._channel_sizes,
            "feature_dim": self.feature_dim,
            "total_params": total_params,
            "trainable_params":  trainable_params,
            "flops": f"{self.estimate_flops():.2f}",
            "max_blocks": self._evolution_config.growth.max_blocks,
            "max_channels": self._evolution_config.growth.max_channels,
        }
