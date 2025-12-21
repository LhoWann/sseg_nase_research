from abc import ABC
from abc import abstractmethod
from typing import Any

import pytorch_lightning as pl
import torch
from torch import Tensor
from torch. optim import Optimizer

from configs.base_config import BaseConfig


class BaseSSLModule(pl. LightningModule, ABC):
    
    def __init__(self, config: BaseConfig):
        super().__init__()
        
        self. save_hyperparameters(config. to_dict())
        self._config = config
    
    @abstractmethod
    def forward(self, x: Tensor) -> Tensor:
        pass
    
    @abstractmethod
    def training_step(self, batch: Any, batch_idx: int) -> Tensor:
        pass
    
    def configure_optimizers(self) -> dict[str, Any]:
        from training.optimizers.optimizer_factory import create_optimizer
        from training.schedulers.lr_scheduler_factory import create_scheduler
        
        optimizer = create_optimizer(
            parameters=self.parameters(),
            optimizer_name="adamw",
            learning_rate=1e-3,
            weight_decay=1e-4,
        )
        
        scheduler = create_scheduler(
            optimizer=optimizer,
            scheduler_name="cosine",
            max_epochs=100,
            warmup_epochs=5,
        )
        
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "epoch",
                "frequency": 1,
            },
        }
    
    def on_train_epoch_start(self) -> None:
        self.log("epoch", float(self.current_epoch), prog_bar=True)
    
    def on_train_epoch_end(self) -> None:
        pass