from typing import Any
from typing import Optional

import torch
import torch.nn.functional as F
from torch import Tensor

from configs.base_config import BaseConfig
from data.augmentations.ssl_augmentation import SSLAugmentation
from models.backbones.evolvable_cnn import EvolvableCNN
from models.heads.projection_head import ProjectionHead
from models.ssl.ema_teacher import EMATeacher
from models.ssl.ssl_losses import CombinedSSLLoss
from training.lightning_modules.base_ssl_module import BaseSSLModule

import warnings
warnings.filterwarnings("ignore", category=UserWarning)

class SSEGModule(BaseSSLModule):
    
    def __init__(self, config: BaseConfig):
        super().__init__(config)
        
        self._backbone = EvolvableCNN(
            seed_config=config.evolution.seed_network,
            evolution_config=config.evolution,
        )
        
        self._projection_head:  Optional[ProjectionHead] = None
        self._teacher: Optional[EMATeacher] = None
        
        self._init_projection_head()
        self._init_teacher()
        
        self._augmentation = SSLAugmentation(
            config=config.ssl.augmentation,
            image_size=config.curriculum.image_size,
        )
        
        self._ssl_loss = CombinedSSLLoss(
            temperature=config.ssl.contrastive.temperature,
            distillation_weight=config.ssl.distillation.distillation_weight,
            distillation_type=config.ssl.distillation.distillation_loss,
        )
        
        self._ssl_loss_history:  list[float] = []
        self._distillation_loss_history: list[float] = []
    
    def _init_projection_head(self) -> None:
        self._projection_head = ProjectionHead(
            input_dim=self._backbone.feature_dim,
            config=self._config.ssl.projection,
        )
    
    def _init_teacher(self) -> None:
        self._teacher = EMATeacher(
            student=self._backbone,
            decay=self._config.ssl.distillation.ema_decay,
        )
    
    def forward(self, x: Tensor) -> Tensor:
        return self._backbone(x)
    
    def training_step(self, batch:  Tensor, batch_idx: int) -> Tensor:
        view1_list = []
        view2_list = []
        
        for image in batch:
            view1, view2 = self._augmentation(image)
            view1_list.append(view1)
            view2_list.append(view2)
        
        view1 = torch.stack(view1_list)
        view2 = torch.stack(view2_list)
        
        features1 = self._backbone(view1)
        features2 = self._backbone(view2)
        
        z1 = self._projection_head(features1)
        z2 = self._projection_head(features2)
        
        with torch.no_grad():
            teacher_features = self._teacher(view1)
        
        total_loss, loss_components = self._ssl_loss(
            z1, z2, features1, teacher_features
        )
        
        if self.current_epoch % self._config.ssl.distillation.update_interval == 0:
            self._teacher.update(self._backbone)
        
        self._ssl_loss_history.append(loss_components["contrastive"])
        self._distillation_loss_history.append(loss_components["distillation"])
        
        self.log_dict(
            {
                "train/ssl_loss": loss_components["contrastive"],
                "train/distillation_loss": loss_components["distillation"],
                "train/total_loss": loss_components["total"],
                "train/num_blocks": float(self._backbone.num_blocks),
                "train/feature_dim": float(self._backbone.feature_dim),
            },
            prog_bar=True,
            sync_dist=True,
        )
        
        return total_loss
    
    def configure_optimizers(self) -> dict[str, Any]:
        from training.optimizers.optimizer_factory import create_optimizer
        from training.schedulers.lr_scheduler_factory import create_scheduler
        
        params = list(self._backbone.parameters()) + list(
            self._projection_head.parameters()
        )
        
        optimizer = create_optimizer(
            parameters=params,
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
    
    def evolve_network(self, mutation_type: str, target_idx: Optional[int] = None) -> bool:
        from models.evolution.evolution_operators import EvolutionOperators
        
        operators = EvolutionOperators(self._backbone, self._config.evolution)
        
        if mutation_type == "grow": 
            result = operators.apply_grow()
        elif mutation_type == "widen" and target_idx is not None: 
            result = operators.apply_widen(target_idx)
        else:
            return False
        
        if result.success:
            self._init_projection_head()
            self._teacher.synchronize_architecture(self._backbone)
        
        return result.success
    
    def get_loss_history(self) -> tuple[list[float], list[float]]: 
        return self._ssl_loss_history.copy(), self._distillation_loss_history.copy()
    
    def clear_loss_history(self) -> None:
        self._ssl_loss_history.clear()
        self._distillation_loss_history.clear()
    
    @property
    def backbone(self) -> EvolvableCNN:
        return self._backbone