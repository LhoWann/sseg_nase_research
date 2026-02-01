from typing import Any
from typing import Optional
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from configs.base_config import BaseConfig
from data.augmentations.ssl_augmentation import SSLAugmentation
from models.backbones.evolvable_cnn import EvolvableCNN
from models.heads.projection_head import ProjectionHead
from models.heads.rotation_head import RotationHead
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
        self._projection_head = self._projection_head.to(self._backbone._blocks[0].conv1.weight.device)
        self._rotation_head: Optional[RotationHead] = None
        if self._config.ssl.rotation.enabled:
            self._init_rotation_head()
            self._rotation_head = self._rotation_head.to(self._backbone._blocks[0].conv1.weight.device)
        self._init_teacher()
        self._augmentation = SSLAugmentation(
            config=config.ssl.augmentation,
            image_size=config.curriculum.image_size,
        )
        self._ssl_loss = CombinedSSLLoss(
            temperature=config.ssl.contrastive.temperature,
            distillation_weight=config.ssl.distillation.distillation_weight,
            distillation_type=config.ssl.distillation.distillation_loss,
            rotation_weight=config.ssl.rotation.weight,
        )
        self._ssl_loss_history:  list[float] = []
        self._distillation_loss_history: list[float] = []
    def _init_projection_head(self) -> None:
        self._projection_head = ProjectionHead(
            input_dim=self._backbone.feature_dim,
            config=self._config.ssl.projection,
        )
    def _init_rotation_head(self) -> None:
        if self._config.ssl.rotation.enabled:
            self._rotation_head = RotationHead(
                input_dim=self._backbone.feature_dim,
                num_classes=4
            )
    def _init_teacher(self) -> None:
        student_model = nn.Sequential(
            self._backbone,
            self._projection_head
        )
        self._teacher = EMATeacher(
            student=student_model,
            decay=self._config.ssl.distillation.ema_decay,
        )
    def forward(self, x: Tensor) -> Tensor:
        return self._backbone(x)
    def get_rotation_loss(self, images: Tensor) -> Tensor:
        batch_size = images.size(0)
        x_0 = images
        x_90 = torch.rot90(images, 1, [2, 3])
        x_180 = torch.rot90(images, 2, [2, 3])
        x_270 = torch.rot90(images, 3, [2, 3])
        x_rotated = torch.cat([x_0, x_90, x_180, x_270], dim=0)
        y_0 = torch.zeros(batch_size, dtype=torch.long, device=images.device)
        y_90 = torch.ones(batch_size, dtype=torch.long, device=images.device)
        y_180 = 2 * torch.ones(batch_size, dtype=torch.long, device=images.device)
        y_270 = 3 * torch.ones(batch_size, dtype=torch.long, device=images.device)
        y_rotated = torch.cat([y_0, y_90, y_180, y_270], dim=0)
        feat_rotated = self._backbone(x_rotated)
        rot_logits = self._rotation_head(feat_rotated)
        loss = F.cross_entropy(rot_logits, y_rotated)
        return loss
    def training_step(self, batch: Tensor, batch_idx: int) -> Tensor:
        crops_list = []
        for image in batch:
            crops = self._augmentation(image)
            crops_list.append(crops)
        n_crops = len(crops_list[0])
        input_views = []
        for i in range(n_crops):
             input_views.append(torch.stack([crops[i] for crops in crops_list]))
        student_outputs = []
        teacher_outputs = []
        for view in input_views:
             feat = self._backbone(view)
             student_outputs.append(self._projection_head(feat))
        with torch.no_grad():
             for view in input_views[:2]:
                  teacher_out = self._teacher(view)
                  teacher_outputs.append(teacher_out)
        if not hasattr(self, "_dino_loss_fn"):
             from models.ssl.dino_loss import DINOLoss
             self._dino_loss_fn = DINOLoss(out_dim=self._config.ssl.projection.output_dim).to(self.device)
        total_loss = self._dino_loss_fn(student_outputs, teacher_outputs, self.current_epoch)
        loss_components = {"dino": total_loss.item(), "total": total_loss.item()}
        if self._rotation_head is not None:
             rot_loss = self.get_rotation_loss(input_views[0])
             total_loss += self._config.ssl.rotation.weight * rot_loss
             loss_components["rotation"] = rot_loss.item()
             loss_components["total"] = total_loss.item()
        if self.current_epoch % self._config.ssl.distillation.update_interval == 0:
            current_student = nn.Sequential(self._backbone, self._projection_head)
            self._teacher.update(current_student) 
        self._ssl_loss_history.append(loss_components["dino"])
        self.log_dict(
            {
                "ssl/dino_loss": loss_components["dino"],
                "ssl/rotation_loss": loss_components.get("rotation", 0.0),
                "train/total_loss": loss_components["total"],
                "network/blocks": float(self._backbone.num_blocks),
                "network/features": float(self._backbone.feature_dim),
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
        if self._rotation_head is not None:
            params.extend(list(self._rotation_head.parameters()))
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
            self._projection_head = self._projection_head.to(self._backbone._blocks[0].conv1.weight.device)
            if self._rotation_head is not None:
                 self._init_rotation_head()
                 self._rotation_head = self._rotation_head.to(self._backbone._blocks[0].conv1.weight.device)
            
            current_student = nn.Sequential(self._backbone, self._projection_head)
            self._teacher.synchronize_architecture(current_student)
        return result.success
    def get_loss_history(self) -> tuple[list[float], list[float]]: 
        return self._ssl_loss_history.copy(), self._distillation_loss_history.copy()
    def clear_loss_history(self) -> None:
        self._ssl_loss_history.clear()
        self._distillation_loss_history.clear()
    @property
    def backbone(self) -> EvolvableCNN:
        return self._backbone
