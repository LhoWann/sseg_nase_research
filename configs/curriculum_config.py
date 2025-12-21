from dataclasses import dataclass
from dataclasses import field
from enum import IntEnum
from typing import Literal


class CurriculumLevel(IntEnum):
    BASIC = 1
    TEXTURE = 2
    OBJECT = 3
    ADVERSARIAL = 4


@dataclass(frozen=True)
class LevelSpec:
    name: str
    num_samples: int
    complexity_min: float
    complexity_max: float
    description: str
    
    def __post_init__(self) -> None:
        if self.num_samples <= 0:
            raise ValueError("num_samples must be positive")
        
        if not 0.0 <= self.complexity_min <= self.complexity_max <= 1.0:
            raise ValueError(
                "complexity must satisfy: 0 <= min <= max <= 1"
            )


@dataclass
class CurriculumConfig: 
    image_size: int = 84
    transition_strategy: Literal["hard", "gradual"] = "gradual"
    gradual_mixing_ratio: float = 0.2
    
    level_specs:  dict[CurriculumLevel, LevelSpec] = field(
        default_factory=lambda: {
            CurriculumLevel. BASIC: LevelSpec(
                name="basic_shapes",
                num_samples=5000,
                complexity_min=0.0,
                complexity_max=0.25,
                description="Simple geometric shapes with solid colors",
            ),
            CurriculumLevel.TEXTURE:  LevelSpec(
                name="textures",
                num_samples=10000,
                complexity_min=0.25,
                complexity_max=0.50,
                description="Texture patterns without semantic objects",
            ),
            CurriculumLevel.OBJECT: LevelSpec(
                name="objects",
                num_samples=20000,
                complexity_min=0.50,
                complexity_max=0.75,
                description="Complex objects with background variation",
            ),
            CurriculumLevel. ADVERSARIAL: LevelSpec(
                name="adversarial",
                num_samples=10000,
                complexity_min=0.75,
                complexity_max=1.0,
                description="Visually similar, semantically different",
            ),
        }
    )
    
    difficulty_weights: tuple[float, float, float] = (0.4, 0.3, 0.3)
    
    def __post_init__(self) -> None:
        if self.image_size <= 0:
            raise ValueError("image_size must be positive")
        
        if not 0.0 <= self.gradual_mixing_ratio <= 1.0:
            raise ValueError("gradual_mixing_ratio must be in [0, 1]")
        
        weight_sum = sum(self.difficulty_weights)
        if abs(weight_sum - 1.0) > 1e-6:
            raise ValueError(
                f"difficulty_weights must sum to 1.0, got {weight_sum}"
            )
    
    def get_level_spec(self, level: CurriculumLevel) -> LevelSpec:
        return self.level_specs[level]
    
    @property
    def total_samples(self) -> int:
        return sum(spec.num_samples for spec in self.level_specs.values())