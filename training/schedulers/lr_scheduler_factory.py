from typing import Literal

from torch.optim import Optimizer
from torch.optim. lr_scheduler import CosineAnnealingLR
from torch.optim.lr_scheduler import StepLR
from torch.optim.lr_scheduler import LambdaLR


def create_scheduler(
    optimizer:  Optimizer,
    scheduler_name: Literal["cosine", "step", "linear_warmup"] = "cosine",
    max_epochs: int = 100,
    warmup_epochs: int = 5,
    min_lr: float = 1e-6,
    step_size: int = 30,
    gamma: float = 0.1,
):
    if scheduler_name == "cosine":
        return CosineAnnealingLR(optimizer, T_max=max_epochs, eta_min=min_lr)
    
    elif scheduler_name == "step":
        return StepLR(optimizer, step_size=step_size, gamma=gamma)
    
    elif scheduler_name == "linear_warmup": 
        
        def warmup_lambda(epoch:  int) -> float:
            if epoch < warmup_epochs: 
                return float(epoch) / float(max(1, warmup_epochs))
            return 1.0
        
        return LambdaLR(optimizer, lr_lambda=warmup_lambda)
    
    else:
        raise ValueError(f"Unknown scheduler:  {scheduler_name}")