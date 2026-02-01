from dataclasses import dataclass
from typing import Literal

@dataclass(frozen=True)
class TrainingConfig:
    batch_size: int = 16
    learning_rate: float = 0.001
    weight_decay: float = 0.0001
    optimizer: Literal["adam", "adamw", "sgd"] = "adamw"
    scheduler: Literal["cosine", "step", "none"] = "cosine"
    warmup_epochs: int = 5
    min_lr: float = 0.00001
    dropout: float = 0.1
    early_stopping_patience: int = 10
    def __post_init__(self) -> None:
        if self.learning_rate <= 0:
            raise ValueError("learning_rate must be positive")
        if self.batch_size <= 0:
            raise ValueError("batch_size must be positive")
