import torch
from torch import Tensor

from configs.evolution_config import NASEConfig


class MaskGenerator:
    
    def __init__(self, config: NASEConfig):
        self._config = config
    
    def generate_complementary_masks(
        self, importance_scores: dict[str, Tensor]
    ) -> tuple[dict[str, Tensor], dict[str, Tensor]]:
        positive_masks = {}
        negative_masks = {}
        
        for name, importance in importance_scores.items():
            threshold = torch.quantile(
                importance.flatten().float(), self._config.sparsity_ratio
            )
            
            positive_mask = (importance >= threshold).float()
            negative_mask = (importance < threshold).float()
            
            min_active = max(
                int(importance.numel() * 0.1), self._config.min_channels_per_layer
            )
            
            if positive_mask.sum() < min_active:
                flat_importance = importance.flatten()
                if flat_importance.numel() < min_active:
                    top_k_indices = torch.arange(flat_importance.numel(), device=flat_importance.device)
                else:
                    top_k_indices = torch.topk(flat_importance, min_active).indices
                positive_mask_flat = positive_mask.flatten()
                positive_mask_flat[top_k_indices] = 1.0
                positive_mask = positive_mask_flat.view(importance.shape)
                negative_mask = 1.0 - positive_mask
            
            positive_masks[name] = positive_mask
            negative_masks[name] = negative_mask
        
        return positive_masks, negative_masks