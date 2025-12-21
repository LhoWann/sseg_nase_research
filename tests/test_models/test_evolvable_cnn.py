import pytest
import torch
import torch.nn as nn
from dataclasses import dataclass
from typing import Literal


@dataclass(frozen=True)
class SeedNetworkConfig:
    architecture: str = "cnn"
    initial_channels: int = 16
    initial_blocks: int = 3
    kernel_size: int = 3
    activation: Literal["relu", "gelu"] = "relu"
    use_batch_norm: bool = True
    use_pooling: bool = True


@dataclass(frozen=True)
class GrowthConfig:
    max_blocks: int = 12
    max_channels: int = 256
    channel_expansion_ratio: float = 1.5


@dataclass
class EvolutionConfig:
    seed_network: SeedNetworkConfig = None
    growth: GrowthConfig = None
    
    def __post_init__(self):
        if self.seed_network is None: 
            object.__setattr__(self, "seed_network", SeedNetworkConfig())
        if self.growth is None: 
            object.__setattr__(self, "growth", GrowthConfig())


class ConvBlock(nn. Module):
    
    def __init__(
        self,
        in_channels:  int,
        out_channels: int,
        kernel_size: int = 3,
        activation: str = "relu",
        use_batch_norm: bool = True,
        use_pooling:  bool = True,
    ):
        super().__init__()
        
        padding = kernel_size // 2
        
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            padding=padding,
            bias=not use_batch_norm,
        )
        
        self.bn = (
            nn.BatchNorm2d(out_channels) if use_batch_norm else nn.Identity()
        )
        
        if activation == "relu": 
            self.activation = nn.ReLU(inplace=True)
        else:
            self.activation = nn. GELU()
        
        self.pool = nn.MaxPool2d(2, 2) if use_pooling else nn.Identity()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        x = self.bn(x)
        x = self.activation(x)
        x = self.pool(x)
        return x


class EvolvableCNN(nn.Module):
    
    def __init__(
        self,
        seed_config: SeedNetworkConfig,
        evolution_config: EvolutionConfig,
    ):
        super().__init__()
        
        self._seed_config = seed_config
        self._evolution_config = evolution_config
        
        self._blocks = nn.ModuleList()
        self._channel_sizes:  list = []
        
        self._build_seed_network()
        
        self.global_pool = nn.AdaptiveAvgPool2d(1)
    
    def _build_seed_network(self) -> None:
        in_channels = 3
        current_channels = self._seed_config.initial_channels
        
        for _ in range(self._seed_config.initial_blocks):
            block = ConvBlock(
                in_channels=in_channels,
                out_channels=current_channels,
                kernel_size=self._seed_config.kernel_size,
                activation=self._seed_config.activation,
                use_batch_norm=self._seed_config.use_batch_norm,
                use_pooling=self._seed_config.use_pooling,
            )
            
            self._blocks.append(block)
            self._channel_sizes.append(current_channels)
            
            in_channels = current_channels
            current_channels = min(current_channels * 2, 256)
    
    def grow(self, out_channels: int = None) -> bool:
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
        )
        
        self._initialize_identity(new_block, in_channels, out_channels)
        
        self._blocks.append(new_block)
        self._channel_sizes.append(out_channels)
        
        return True
    
    def _initialize_identity(
        self, block:  ConvBlock, in_channels: int, out_channels: int
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
        
        new_block = ConvBlock(
            in_channels=in_channels,
            out_channels=new_channels,
            kernel_size=self._seed_config.kernel_size,
            activation=self._seed_config.activation,
            use_batch_norm=self._seed_config.use_batch_norm,
            use_pooling=self._seed_config.use_pooling,
        )
        
        with torch.no_grad():
            new_block.conv.weight[: old_channels] = old_block.conv.weight
            
            if self._seed_config.use_batch_norm:
                new_block.bn.weight[:old_channels] = old_block.bn.weight
                new_block.bn.bias[:old_channels] = old_block.bn.bias
                new_block.bn.running_mean[: old_channels] = old_block.bn.running_mean
                new_block.bn.running_var[:old_channels] = old_block.bn.running_var
        
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
        
        new_conv = nn.Conv2d(
            new_in_channels,
            out_channels,
            kernel_size=old_conv.kernel_size,
            stride=old_conv.stride,
            padding=old_conv.padding,
            bias=old_conv.bias is not None,
        )
        
        with torch.no_grad():
            new_conv.weight[: , :old_in_channels] = old_conv.weight
            
            if old_conv.bias is not None:
                new_conv.bias.copy_(old_conv.bias)
        
        next_block.conv = new_conv
    
    @property
    def feature_dim(self) -> int:
        return self._channel_sizes[-1] if self._channel_sizes else 0
    
    @property
    def num_blocks(self) -> int:
        return len(self._blocks)
    
    @property
    def channel_sizes(self) -> list:
        return self._channel_sizes.copy()
    
    def forward(self, x:  torch.Tensor) -> torch.Tensor:
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
            "trainable_params": trainable_params,
            "max_blocks": self._evolution_config.growth.max_blocks,
            "max_channels": self._evolution_config.growth.max_channels,
        }


class TestConvBlock: 
    
    def test_conv_block_output_shape(self) -> None:
        block = ConvBlock(
            in_channels=3,
            out_channels=16,
            kernel_size=3,
            activation="relu",
            use_batch_norm=True,
            use_pooling=True,
        )
        
        x = torch.randn(2, 3, 84, 84)
        output = block(x)
        
        assert output.shape == (2, 16, 42, 42)
    
    def test_conv_block_without_pooling(self) -> None:
        block = ConvBlock(
            in_channels=3,
            out_channels=16,
            kernel_size=3,
            activation="relu",
            use_batch_norm=True,
            use_pooling=False,
        )
        
        x = torch.randn(2, 3, 84, 84)
        output = block(x)
        
        assert output.shape == (2, 16, 84, 84)
    
    def test_conv_block_gelu_activation(self) -> None:
        block = ConvBlock(
            in_channels=3,
            out_channels=16,
            kernel_size=3,
            activation="gelu",
            use_batch_norm=True,
            use_pooling=True,
        )
        
        x = torch.randn(2, 3, 84, 84)
        output = block(x)
        
        assert output.shape == (2, 16, 42, 42)


class TestEvolvableCNN:
    
    @pytest.fixture
    def seed_config(self) -> SeedNetworkConfig: 
        return SeedNetworkConfig(
            initial_channels=16,
            initial_blocks=3,
            kernel_size=3,
            activation="relu",
        )
    
    @pytest.fixture
    def evolution_config(self, seed_config: SeedNetworkConfig) -> EvolutionConfig:
        growth_config = GrowthConfig(
            max_blocks=8,
            max_channels=128,
            channel_expansion_ratio=1.5,
        )
        return EvolutionConfig(seed_network=seed_config, growth=growth_config)
    
    @pytest.fixture
    def model(
        self,
        seed_config: SeedNetworkConfig,
        evolution_config: EvolutionConfig,
    ) -> EvolvableCNN: 
        return EvolvableCNN(
            seed_config=seed_config,
            evolution_config=evolution_config,
        )
    
    def test_initial_architecture(self, model: EvolvableCNN) -> None:
        assert model.num_blocks == 3
        assert len(model.channel_sizes) == 3
    
    def test_forward_pass_returns_correct_shape(
        self, model:  EvolvableCNN
    ) -> None:
        x = torch.randn(4, 3, 84, 84)
        output = model(x)
        
        assert output.shape == (4, model.feature_dim)
    
    def test_feature_dim_equals_last_channel_size(
        self, model: EvolvableCNN
    ) -> None:
        assert model.feature_dim == model.channel_sizes[-1]
    
    def test_grow_adds_block(self, model:  EvolvableCNN) -> None:
        initial_blocks = model.num_blocks
        
        success = model.grow()
        
        assert success is True
        assert model.num_blocks == initial_blocks + 1
    
    def test_grow_fails_at_max_blocks(
        self, seed_config: SeedNetworkConfig
    ) -> None:
        growth_config = GrowthConfig(
            max_blocks=3,
            max_channels=128,
        )
        evolution_config = EvolutionConfig(
            seed_network=seed_config,
            growth=growth_config,
        )
        model = EvolvableCNN(
            seed_config=seed_config,
            evolution_config=evolution_config,
        )
        
        success = model.grow()
        
        assert success is False
        assert model.num_blocks == 3
    
    def test_grow_preserves_forward_pass(self, model: EvolvableCNN) -> None:
        x = torch.randn(4, 3, 84, 84)
        
        model.grow()
        output = model(x)
        
        assert output.shape[0] == 4
        assert output.shape[1] == model.feature_dim
    
    def test_widen_increases_channels(self, model: EvolvableCNN) -> None:
        initial_channels = model.channel_sizes[0]
        
        success = model.widen(block_idx=0)
        
        assert success is True
        assert model.channel_sizes[0] > initial_channels
    
    def test_widen_fails_at_max_channels(
        self, seed_config: SeedNetworkConfig
    ) -> None:
        growth_config = GrowthConfig(
            max_blocks=8,
            max_channels=16,
        )
        evolution_config = EvolutionConfig(
            seed_network=seed_config,
            growth=growth_config,
        )
        model = EvolvableCNN(
            seed_config=seed_config,
            evolution_config=evolution_config,
        )
        
        success = model.widen(block_idx=0)
        
        assert success is False
    
    def test_widen_preserves_forward_pass(self, model: EvolvableCNN) -> None:
        x = torch.randn(4, 3, 84, 84)
        
        model.widen(block_idx=0)
        output = model(x)
        
        assert output.shape[0] == 4
    
    def test_widen_invalid_block_index_returns_false(
        self, model: EvolvableCNN
    ) -> None:
        success = model.widen(block_idx=100)
        
        assert success is False
    
    def test_get_architecture_summary_returns_dict(
        self, model: EvolvableCNN
    ) -> None:
        summary = model.get_architecture_summary()
        
        assert isinstance(summary, dict)
        assert "num_blocks" in summary
        assert "channel_progression" in summary
        assert "feature_dim" in summary
        assert "total_params" in summary
    
    def test_architecture_summary_values_are_correct(
        self, model: EvolvableCNN
    ) -> None:
        summary = model.get_architecture_summary()
        
        assert summary["num_blocks"] == model.num_blocks
        assert summary["feature_dim"] == model.feature_dim
        assert summary["channel_progression"] == model.channel_sizes
    
    def test_multiple_grow_operations(self, model:  EvolvableCNN) -> None:
        initial_blocks = model.num_blocks
        
        model.grow()
        model.grow()
        model.grow()
        
        assert model.num_blocks == initial_blocks + 3
    
    def test_grow_then_forward_maintains_consistency(
        self, model: EvolvableCNN
    ) -> None:
        x = torch.randn(4, 3, 84, 84)
        
        for _ in range(3):
            model.grow()
            output = model(x)
            assert output.shape == (4, model.feature_dim)