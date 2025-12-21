import pytest
import torch
import torch.nn as nn
from dataclasses import dataclass
from dataclasses import field
from enum import Enum
from enum import auto
from pathlib import Path
from typing import Optional
import json


class MutationType(Enum):
    GROW = auto()
    WIDEN = auto()
    PRUNE = auto()
    NONE = auto()


@dataclass(frozen=True)
class MutationResult:
    mutation_type: MutationType
    target_layer: Optional[int]
    success: bool
    old_num_blocks: int
    new_num_blocks: int
    old_num_params: int
    new_num_params: int


@dataclass(frozen=True)
class SeedNetworkConfig: 
    initial_channels: int = 16
    initial_blocks: int = 3
    kernel_size: int = 3
    activation: str = "relu"
    use_batch_norm: bool = True
    use_pooling: bool = True


@dataclass(frozen=True)
class GrowthConfig:
    max_blocks: int = 12
    max_channels: int = 256
    channel_expansion_ratio: float = 1.5
    sensitivity_method: str = "taylor"


@dataclass
class EvolutionConfig:
    seed_network: SeedNetworkConfig = None
    growth:  GrowthConfig = None
    
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
        use_batch_norm:  bool = True,
        use_pooling: bool = True,
    ):
        super().__init__()
        
        padding = kernel_size // 2
        self.conv = nn. Conv2d(
            in_channels, out_channels, kernel_size,
            padding=padding, bias=not use_batch_norm
        )
        self.bn = nn.BatchNorm2d(out_channels) if use_batch_norm else nn.Identity()
        self.activation = nn.ReLU(inplace=True) if activation == "relu" else nn. GELU()
        self.pool = nn.MaxPool2d(2, 2) if use_pooling else nn.Identity()
    
    def forward(self, x: torch. Tensor) -> torch.Tensor:
        return self.pool(self.activation(self.bn(self.conv(x))))


class EvolvableCNN(nn.Module):
    
    def __init__(self, seed_config:  SeedNetworkConfig, evolution_config:  EvolutionConfig):
        super().__init__()
        self._seed_config = seed_config
        self._evolution_config = evolution_config
        self._blocks = nn. ModuleList()
        self._channel_sizes = []
        self._build_seed_network()
        self. global_pool = nn.AdaptiveAvgPool2d(1)
    
    def _build_seed_network(self) -> None:
        in_ch = 3
        out_ch = self._seed_config. initial_channels
        for _ in range(self._seed_config. initial_blocks):
            self._blocks.append(ConvBlock(in_ch, out_ch))
            self._channel_sizes.append(out_ch)
            in_ch = out_ch
            out_ch = min(out_ch * 2, 256)
    
    def grow(self) -> bool:
        if len(self._blocks) >= self._evolution_config.growth.max_blocks:
            return False
        in_ch = self._channel_sizes[-1]
        out_ch = min(
            int(in_ch * self._evolution_config. growth.channel_expansion_ratio),
            self._evolution_config.growth.max_channels,
        )
        self._blocks.append(ConvBlock(in_ch, out_ch))
        self._channel_sizes. append(out_ch)
        return True
    
    def widen(self, block_idx: int) -> bool:
        if block_idx >= len(self._blocks):
            return False
        old_ch = self._channel_sizes[block_idx]
        new_ch = min(
            int(old_ch * self._evolution_config.growth.channel_expansion_ratio),
            self._evolution_config. growth.max_channels,
        )
        if new_ch == old_ch:
            return False
        self._channel_sizes[block_idx] = new_ch
        return True
    
    @property
    def num_blocks(self) -> int:
        return len(self._blocks)
    
    @property
    def channel_sizes(self) -> list:
        return self._channel_sizes.copy()
    
    @property
    def feature_dim(self) -> int:
        return self._channel_sizes[-1]
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for block in self._blocks:
            x = block(x)
        return self.global_pool(x).flatten(1)
    
    def get_architecture_summary(self) -> dict:
        return {
            "num_blocks": self. num_blocks,
            "channel_progression": self._channel_sizes,
            "feature_dim": self.feature_dim,
            "total_params": sum(p.numel() for p in self.parameters()),
        }


class EvolutionOperators:
    
    def __init__(self, model: EvolvableCNN, config: EvolutionConfig):
        self._model = model
        self._config = config
    
    def apply_grow(self, out_channels: Optional[int] = None) -> MutationResult: 
        old_summary = self._model. get_architecture_summary()
        success = self._model. grow()
        new_summary = self._model.get_architecture_summary()
        
        return MutationResult(
            mutation_type=MutationType. GROW,
            target_layer=None,
            success=success,
            old_num_blocks=old_summary["num_blocks"],
            new_num_blocks=new_summary["num_blocks"],
            old_num_params=old_summary["total_params"],
            new_num_params=new_summary["total_params"],
        )
    
    def apply_widen(self, block_idx:  int) -> MutationResult:
        old_summary = self._model.get_architecture_summary()
        success = self._model.widen(block_idx)
        new_summary = self._model.get_architecture_summary()
        
        return MutationResult(
            mutation_type=MutationType. WIDEN,
            target_layer=block_idx,
            success=success,
            old_num_blocks=old_summary["num_blocks"],
            new_num_blocks=new_summary["num_blocks"],
            old_num_params=old_summary["total_params"],
            new_num_params=new_summary["total_params"],
        )
    
    def can_grow(self) -> bool:
        return self._model. num_blocks < self._config.growth. max_blocks
    
    def can_widen(self, block_idx: int) -> bool:
        if block_idx >= self._model.num_blocks:
            return False
        current_channels = self._model. channel_sizes[block_idx]
        return current_channels < self._config.growth. max_channels


@dataclass(frozen=True)
class LayerSensitivity:
    layer_idx: int
    sensitivity_score: float
    current_channels: int
    can_widen: bool


class MutationSelector:
    
    def __init__(self, model: EvolvableCNN, config: EvolutionConfig):
        self._model = model
        self._config = config
    
    def compute_layer_sensitivities(self) -> list:
        sensitivities = []
        
        for idx, block in enumerate(self._model._blocks):
            sensitivity_score = self._compute_block_sensitivity(block)
            current_channels = self._model.channel_sizes[idx]
            can_widen = current_channels < self._config.growth.max_channels
            
            sensitivities. append(
                LayerSensitivity(
                    layer_idx=idx,
                    sensitivity_score=sensitivity_score,
                    current_channels=current_channels,
                    can_widen=can_widen,
                )
            )
        
        return sorted(sensitivities, key=lambda s: s.sensitivity_score, reverse=True)
    
    def _compute_block_sensitivity(self, block) -> float:
        total_sensitivity = 0.0
        param_count = 0
        
        for param in block.parameters():
            if param. grad is None:
                continue
            sensitivity = (param.data * param.grad).abs().sum().item()
            total_sensitivity += sensitivity
            param_count += 1
        
        return total_sensitivity / max(param_count, 1)
    
    def select_mutation(self, sensitivities: list) -> tuple:
        can_grow = self._model.num_blocks < self._config.growth.max_blocks
        
        if can_grow:
            return MutationType.GROW, None
        
        for sensitivity in sensitivities:
            if sensitivity.can_widen:
                return MutationType.WIDEN, sensitivity.layer_idx
        
        return MutationType.NONE, None


@dataclass
class MutationRecord:
    epoch: int
    level: int
    mutation_type: str
    target_layer: Optional[int]
    num_blocks_before: int
    num_blocks_after: int
    num_params_before: int
    num_params_after:  int
    ssl_loss_before: float
    ssl_loss_after: Optional[float] = None


@dataclass
class ArchitectureTracker:
    mutation_history: list = field(default_factory=list)
    architecture_snapshots: list = field(default_factory=list)
    
    def record_mutation(
        self,
        epoch: int,
        level:  int,
        mutation_type: MutationType,
        target_layer: Optional[int],
        num_blocks_before: int,
        num_blocks_after: int,
        num_params_before: int,
        num_params_after: int,
        ssl_loss_before: float,
    ) -> None:
        record = MutationRecord(
            epoch=epoch,
            level=level,
            mutation_type=mutation_type.name,
            target_layer=target_layer,
            num_blocks_before=num_blocks_before,
            num_blocks_after=num_blocks_after,
            num_params_before=num_params_before,
            num_params_after=num_params_after,
            ssl_loss_before=ssl_loss_before,
        )
        self.mutation_history.append(record)
    
    def record_architecture(self, epoch: int, architecture_summary: dict) -> None:
        snapshot = {"epoch": epoch, **architecture_summary}
        self.architecture_snapshots.append(snapshot)
    
    def save(self, filepath: Path) -> None:
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        data = {
            "mutation_history": [
                {
                    "epoch": m.epoch,
                    "level": m.level,
                    "mutation_