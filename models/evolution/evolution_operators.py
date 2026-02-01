from dataclasses import dataclass
from enum import Enum
from enum import auto
from typing import Optional
from torch import Tensor
from configs.evolution_config import EvolutionConfig
from models.backbones.evolvable_cnn import EvolvableCNN
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
class EvolutionOperators: 
    def __init__(self, model: EvolvableCNN, config: EvolutionConfig):
        self._model = model
        self._config = config
    def apply_grow(self, out_channels: Optional[int] = None) -> MutationResult:
        old_summary = self._model.get_architecture_summary()
        success = self._model.grow(out_channels)
        new_summary = self._model.get_architecture_summary()
        return MutationResult(
            mutation_type=MutationType.GROW,
            target_layer=None,
            success=success,
            old_num_blocks=old_summary["num_blocks"],
            new_num_blocks=new_summary["num_blocks"],
            old_num_params=old_summary["total_params"],
            new_num_params=new_summary["total_params"],
        )
    def apply_widen(self, block_idx: int) -> MutationResult:
        old_summary = self._model.get_architecture_summary()
        success = self._model.widen(block_idx)
        new_summary = self._model.get_architecture_summary()
        return MutationResult(
            mutation_type=MutationType.WIDEN,
            target_layer=block_idx,
            success=success,
            old_num_blocks=old_summary["num_blocks"],
            new_num_blocks=new_summary["num_blocks"],
            old_num_params=old_summary["total_params"],
            new_num_params=new_summary["total_params"],
        )
    def can_grow(self) -> bool:
        return self._model.num_blocks < self._config.growth.max_blocks
    def can_widen(self, block_idx: int) -> bool:
        if block_idx >= self._model.num_blocks:
            return False
        current_channels = self._model.channel_sizes[block_idx]
        return current_channels < self._config.growth.max_channels
