from dataclasses import dataclass
from typing import Literal


@dataclass(frozen=True)
class AugmentationConfig:
    crop_scale_min: float = 0.2
    crop_scale_max:  float = 1.0
    horizontal_flip_prob: float = 0.5
    color_jitter_strength: float = 0.4
    grayscale_prob: float = 0.2
    gaussian_blur_prob: float = 0.5
    gaussian_blur_kernel_size: int = 9
    
    def __post_init__(self) -> None:
        if not 0.0 < self.crop_scale_min <= self.crop_scale_max <= 1.0:
            raise ValueError(
                "crop_scale must satisfy: 0 < min <= max <= 1"
            )
        
        if not 0.0 <= self.horizontal_flip_prob <= 1.0:
            raise ValueError("horizontal_flip_prob must be in [0, 1]")
        
        if self.gaussian_blur_kernel_size % 2 == 0:
            raise ValueError("gaussian_blur_kernel_size must be odd")


@dataclass(frozen=True)
class ContrastiveLossConfig:
    temperature: float = 0.5
    loss_type: Literal["ntxent", "infonce"] = "ntxent"
    normalize_features: bool = True
    
    def __post_init__(self) -> None:
        if self.temperature <= 0:
            raise ValueError("temperature must be positive")


@dataclass(frozen=True)
class DistillationConfig:
    ema_decay: float = 0.999
    distillation_weight: float = 0.5
    distillation_loss:  Literal["mse", "cosine", "kl"] = "mse"
    update_interval:  int = 1
    
    def __post_init__(self) -> None:
        if not 0.0 < self. ema_decay < 1.0:
            raise ValueError("ema_decay must be in (0, 1)")
        
        if self.distillation_weight < 0:
            raise ValueError("distillation_weight must be non-negative")
        
        if self.update_interval <= 0:
            raise ValueError("update_interval must be positive")


@dataclass(frozen=True)
class ProjectionConfig:
    hidden_dim: int = 256
    output_dim: int = 128
    num_layers: int = 2
    use_batch_norm: bool = True
    
    def __post_init__(self) -> None:
        if self.hidden_dim <= 0:
            raise ValueError("hidden_dim must be positive")
        
        if self.output_dim <= 0:
            raise ValueError("output_dim must be positive")
        
        if self.num_layers <= 0:
            raise ValueError("num_layers must be positive")


@dataclass
class SSLConfig:
    augmentation: AugmentationConfig
    contrastive:  ContrastiveLossConfig
    distillation: DistillationConfig
    projection: ProjectionConfig
    
    def __init__(
        self,
        augmentation:  AugmentationConfig = AugmentationConfig(),
        contrastive: ContrastiveLossConfig = ContrastiveLossConfig(),
        distillation: DistillationConfig = DistillationConfig(),
        projection: ProjectionConfig = ProjectionConfig(),
    ):
        object.__setattr__(self, "augmentation", augmentation)
        object.__setattr__(self, "contrastive", contrastive)
        object.__setattr__(self, "distillation", distillation)
        object.__setattr__(self, "projection", projection)