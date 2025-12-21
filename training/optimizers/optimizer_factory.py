from typing import Iterable
from typing import Literal

from torch import nn
from torch.optim import Optimizer
from torch.optim import AdamW
from torch.optim import SGD


def create_optimizer(
    parameters: Iterable[nn.Parameter],
    optimizer_name:  Literal["adamw", "sgd", "adam"] = "adamw",
    learning_rate: float = 1e-3,
    weight_decay: float = 1e-4,
    momentum: float = 0.9,
    betas: tuple[float, float] = (0.9, 0.999),
) -> Optimizer:
    if optimizer_name == "adamw": 
        return AdamW(
            parameters, lr=learning_rate, weight_decay=weight_decay, betas=betas
        )
    
    elif optimizer_name == "sgd":
        return SGD(
            parameters,
            lr=learning_rate,
            weight_decay=weight_decay,
            momentum=momentum,
        )
    
    elif optimizer_name == "adam":
        from torch.optim import Adam
        
        return Adam(parameters, lr=learning_rate, weight_decay=weight_decay, betas=betas)
    
    else:
        raise ValueError(f"Unknown optimizer: {optimizer_name}")