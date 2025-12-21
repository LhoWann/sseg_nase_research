from typing import Literal

import torch
from torch import Tensor
import torch.nn as nn


class ImportanceScorer: 
    
    def __init__(self, method: Literal["magnitude", "gradient", "taylor"] = "taylor"):
        self._method = method
    
    def compute_importance(self, model: nn.Module) -> dict[str, Tensor]:
        importance_scores = {}
        
        for name, param in model.named_parameters():
            if param.grad is None:
                continue
            
            if self._method == "magnitude":
                importance = param.data.abs()
            
            elif self._method == "gradient":
                importance = param.grad.abs()
            
            else: 
                importance = (param.data * param.grad).abs()
            
            importance_scores[name] = importance.detach().cpu()
        
        return importance_scores