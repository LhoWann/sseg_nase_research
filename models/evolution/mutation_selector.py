from dataclasses import dataclass
from typing import Optional
import torch
from torch import Tensor
from configs.evolution_config import EvolutionConfig
from configs.evolution_config import GrowthConfig
from models.backbones.evolvable_cnn import EvolvableCNN
from models.evolution.evolution_operators import MutationType
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
    def compute_layer_sensitivities(self) -> list[LayerSensitivity]: 
        sensitivities = []
        for idx, block in enumerate(self._model._blocks):
            sensitivity_score = self._compute_block_sensitivity(
                block, self._config.growth.sensitivity_method
            )
            current_channels = self._model.channel_sizes[idx]
            can_widen = current_channels < self._config.growth.max_channels
            sensitivities.append(
                LayerSensitivity(
                    layer_idx=idx,
                    sensitivity_score=sensitivity_score,
                    current_channels=current_channels,
                    can_widen=can_widen,
                )
            )
        return sorted(sensitivities, key=lambda s: s.sensitivity_score, reverse=True)
    def _compute_block_sensitivity(self, block, method: str) -> float:
        total_sensitivity = 0.0
        param_count = 0
        for param in block.parameters():
            if param.grad is None:
                continue
            if method == "gradient": 
                sensitivity = torch.sqrt((param.grad ** 2).sum()).item()
            elif method == "taylor":
                sensitivity = torch.sqrt(((param.data * param.grad) ** 2).sum()).item()
            else:
                sensitivity = torch.sqrt((param.grad ** 2).sum()).item()
            total_sensitivity += sensitivity
            param_count += 1
        return total_sensitivity / max(param_count, 1)

    def _compute_fitness(self, mutation_type: MutationType, target_idx: Optional[int]) -> float:
        current_flops = self._model.estimate_flops()
        target_flops = self._config.fitness.target_flops_giga
        alpha = self._config.fitness.alpha_complexity_penalty
        flops_penalty = alpha * (current_flops / target_flops)
        base_score = 1.0
        if mutation_type == MutationType.GROW:
            base_score = 0.8
        elif mutation_type == MutationType.WIDEN:
            base_score = 0.6
        return base_score - flops_penalty

    def select_mutation(
        self, sensitivities: list[LayerSensitivity]
    ) -> tuple[MutationType, Optional[int]]:
        candidates = []
        can_grow = self._model.num_blocks < self._config.growth.max_blocks
        if can_grow:
            fitness = self._compute_fitness(MutationType.GROW, None)
            if fitness > 0:
                candidates.append((MutationType.GROW, None, fitness))
        for sensitivity in sensitivities:
            if sensitivity.can_widen:
                fitness = self._compute_fitness(MutationType.WIDEN, sensitivity.layer_idx)
                weighted_fitness = fitness * (1.0 + sensitivity.sensitivity_score * 0.1)
                if weighted_fitness > 0:
                    candidates.append((MutationType.WIDEN, sensitivity.layer_idx, weighted_fitness))
        if not candidates:
            return MutationType.NONE, None
        candidates.sort(key=lambda x: x[2], reverse=True)
        best = candidates[0]
        return best[0], best[1]
