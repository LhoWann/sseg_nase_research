from dataclasses import dataclass
from typing import Optional

import torch
from torch import Tensor
import torch.nn as nn

from configs.evolution_config import NASEConfig
from models.nase.importance_scorer import ImportanceScorer
from models.nase.mask_generator import MaskGenerator


@dataclass(frozen=True)
class SparsityStatistics:
    total_connections: int
    active_positive_connections: int
    active_negative_connections: int
    sparsity_ratio: float


class SparseRouter:
    
    def __init__(self, config: NASEConfig):
        self._config = config
        self._importance_scorer = ImportanceScorer(config.importance_metric)
        self._mask_generator = MaskGenerator(config)
        
        self._positive_masks:  dict[str, Tensor] = {}
        self._negative_masks: dict[str, Tensor] = {}
    
    def update_masks(self, model: nn.Module) -> None:
        importance_scores = self._importance_scorer.compute_importance(model)
        
        self._positive_masks, self._negative_masks = (
            self._mask_generator.generate_complementary_masks(importance_scores)
        )
    
    def apply_sparse_forward(
        self, model: nn.Module, x: Tensor, use_negative_path: bool = False
    ) -> Tensor:
        masks = self._negative_masks if use_negative_path else self._positive_masks
        
        if not masks:
            return model(x)
        
        original_params = {}
        
        for name, param in model.named_parameters():
            if name in masks:
                original_params[name] = param.data.clone()
                mask_device = masks[name].to(param.device)
                param.data = param.data * mask_device
        
        output = model(x)
        
        for name, original_data in original_params.items():
            dict(model.named_parameters())[name].data = original_data
        
        return output
    
    def get_statistics(self) -> Optional[SparsityStatistics]: 
        if not self._positive_masks: 
            return None
        
        total = sum(mask.numel() for mask in self._positive_masks.values())
        active_pos = sum(mask.sum().item() for mask in self._positive_masks.values())
        active_neg = sum(mask.sum().item() for mask in self._negative_masks.values())
        
        return SparsityStatistics(
            total_connections=total,
            active_positive_connections=int(active_pos),
            active_negative_connections=int(active_neg),
            sparsity_ratio=1.0 - (active_pos / total),
        )
    
    def prune_permanently(self, model: nn.Module) -> None:
        with torch.no_grad():
            for name, param in model.named_parameters():
                if name in self._positive_masks:
                    mask = self._positive_masks[name].to(param.device)
                    param.data = param.data * mask