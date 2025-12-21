from dataclasses import dataclass
from dataclasses import field
from typing import Literal


@dataclass(frozen=True)
class FewShotConfig:
    num_ways: int = 5
    num_shots: tuple[int, ... ] = (1, 5)
    num_queries_per_class: int = 15
    num_episodes:  int = 600
    
    distance_metric: Literal["euclidean", "cosine"] = "cosine"
    normalize_features: bool = True
    
    def __post_init__(self) -> None:
        if self.num_ways <= 0:
            raise ValueError("num_ways must be positive")
        
        if any(shot <= 0 for shot in self. num_shots):
            raise ValueError("all num_shots must be positive")
        
        if self.num_queries_per_class <= 0:
            raise ValueError("num_queries_per_class must be positive")
        
        if self. num_episodes <= 0:
            raise ValueError("num_episodes must be positive")


@dataclass(frozen=True)
class MetricsConfig:
    confidence_level: float = 0.95
    compute_per_class_accuracy: bool = False
    save_predictions: bool = False
    
    def __post_init__(self) -> None:
        if not 0.0 < self.confidence_level < 1.0:
            raise ValueError("confidence_level must be in (0, 1)")


@dataclass(frozen=True)
class DatasetConfig:
    name:  Literal["minimagenet", "cifar_fs", "tiered_imagenet"]
    root_dir: str
    split:  Literal["train", "val", "test"]
    
    def __post_init__(self) -> None:
        if not self.root_dir:
            raise ValueError("root_dir cannot be empty")


@dataclass
class EvaluationConfig: 
    few_shot:  FewShotConfig
    metrics: MetricsConfig
    datasets: list[DatasetConfig] = field(default_factory=list)
    
    adaptation_method: Literal["prototype", "linear", "finetune"] = "prototype"
    adaptation_steps: int = 0
    adaptation_lr: float = 0.001
    
    def __init__(
        self,
        few_shot: FewShotConfig = FewShotConfig(),
        metrics: MetricsConfig = MetricsConfig(),
        datasets: list[DatasetConfig] = None,
        adaptation_method:  Literal[
            "prototype", "linear", "finetune"
        ] = "prototype",
        adaptation_steps: int = 0,
        adaptation_lr: float = 0.001,
    ):
        object.__setattr__(self, "few_shot", few_shot)
        object.__setattr__(self, "metrics", metrics)
        object.__setattr__(self, "datasets", datasets or [])
        object.__setattr__(self, "adaptation_method", adaptation_method)
        object.__setattr__(self, "adaptation_steps", adaptation_steps)
        object.__setattr__(self, "adaptation_lr", adaptation_lr)
    
    def __post_init__(self) -> None:
        if self.adaptation_steps < 0:
            raise ValueError("adaptation_steps must be non-negative")
        
        if self. adaptation_lr <= 0:
            raise ValueError("adaptation_lr must be positive")