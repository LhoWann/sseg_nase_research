from dataclasses import dataclass
from typing import Literal
@dataclass(frozen=True)
class SeedNetworkConfig:
    architecture: Literal["cnn", "tiny_vit"] = "cnn"
    initial_channels: int = 16
    initial_blocks: int = 3
    kernel_size: int = 3
    activation: Literal["relu", "gelu"] = "relu"
    use_batch_norm: bool = True
    use_pooling: bool = True
    def __post_init__(self) -> None:
        if self.initial_channels <= 0:
            raise ValueError("initial_channels must be positive")
        if self.initial_blocks <= 0:
            raise ValueError("initial_blocks must be positive")
        if self.kernel_size % 2 == 0:
            raise ValueError("kernel_size must be odd")
@dataclass(frozen=True)
class GrowthConfig: 
    max_blocks: int = 12
    max_channels: int = 256
    channel_expansion_ratio: float = 1.5
    plateau_window_size: int = 10
    plateau_threshold: float = 1e-4
    distillation_gap_threshold: float = 0.1
    sensitivity_method: Literal["gradient", "taylor", "fisher"] = "taylor"
    def __post_init__(self) -> None:
        if self.max_blocks <= 0:
            raise ValueError("max_blocks must be positive")
        if self.max_channels <= 0:
            raise ValueError("max_channels must be positive")
        if self.channel_expansion_ratio <= 1.0:
            raise ValueError("channel_expansion_ratio must be > 1.0")
        if self.plateau_window_size <= 0:
            raise ValueError("plateau_window_size must be positive")
        if self.plateau_threshold <= 0:
            raise ValueError("plateau_threshold must be positive")
@dataclass(frozen=True)
class NASEConfig: 
    sparsity_ratio: float = 0.3
    pruning_interval_epochs: int = 10
    importance_metric: Literal["magnitude", "gradient", "taylor"] = "taylor"
    min_channels_per_layer: int = 8
    use_complementary_masks: bool = True
    def __post_init__(self) -> None:
        if not 0.0 < self.sparsity_ratio < 1.0:
            raise ValueError("sparsity_ratio must be in (0, 1)")
        if self.pruning_interval_epochs <= 0:
            raise ValueError("pruning_interval_epochs must be positive")
        if self.min_channels_per_layer <= 0:
            raise ValueError("min_channels_per_layer must be positive")
@dataclass(frozen=True)
class FitnessConfig:
    alpha_complexity_penalty: float = 0.1
    target_flops_giga: float = 1.0
    target_params_million: float = 1.0
    def __post_init__(self) -> None:
        if self.alpha_complexity_penalty < 0:
            raise ValueError("alpha_complexity_penalty must be non-negative")
        if self.target_flops_giga <= 0:
            raise ValueError("target_flops_giga must be positive")
        if self.target_params_million <= 0:
            raise ValueError("target_params_million must be positive")
@dataclass
class EvolutionConfig: 
    seed_network: SeedNetworkConfig
    growth: GrowthConfig
    nase: NASEConfig
    fitness: FitnessConfig
    def __init__(
        self,
        seed_network: SeedNetworkConfig = SeedNetworkConfig(),
        growth: GrowthConfig = GrowthConfig(),
        nase: NASEConfig = NASEConfig(),
        fitness: FitnessConfig = FitnessConfig(),
    ):
        object.__setattr__(self, "seed_network", seed_network)
        object.__setattr__(self, "growth", growth)
        object.__setattr__(self, "nase", nase)
        object.__setattr__(self, "fitness", fitness)
