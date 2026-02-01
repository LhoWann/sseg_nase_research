from dataclasses import dataclass
from dataclasses import field
from pathlib import Path
from typing import Optional
from configs.curriculum_config import CurriculumConfig
from configs.evaluation_config import EvaluationConfig
from configs.evolution_config import EvolutionConfig
from configs.hardware_config import HardwareConfig
from configs.ssl_config import SSLConfig
from configs.training_config import TrainingConfig
@dataclass
class PathConfig:
    root: Path
    data: Path
    outputs: Path
    checkpoints: Path
    logs: Path
    def create_directories(self) -> None:
        for path in [
            self.data,
            self.outputs,
            self.checkpoints,
            self.logs,
        ]:
            path.mkdir(parents=True, exist_ok=True)
@dataclass
class BaseConfig: 
    experiment_name: str
    seed: int
    paths: PathConfig
    hardware: HardwareConfig
    curriculum: CurriculumConfig
    evolution: EvolutionConfig
    ssl: SSLConfig
    evaluation: EvaluationConfig
    training: TrainingConfig = None
    resume_from_checkpoint: Optional[Path] = None
    debug_mode: bool = False
    def __post_init__(self) -> None:
        if self.seed < 0:
            raise ValueError("seed must be non-negative")
        self.paths.create_directories()
    @classmethod
    def from_experiment_name(
        cls,
        experiment_name: str,
        root_dir: Path = Path("./"),
        hardware_config: Optional[HardwareConfig] = None,
    ) -> "BaseConfig":
        from utils.environment import EnvironmentHelper
        env_paths = EnvironmentHelper.get_paths(experiment_name)
        
        # If root_dir is explicitly provided and NOT default "./", usually we might want to respect it.
        # But EnvironmentHelper logic is robust for cloud envs. 
        # We can treat EnvironmentHelper as the source of truth for defaults.
        
        paths = PathConfig(
            root=env_paths["root"],
            data=env_paths["data"],
            outputs=env_paths["outputs"],
            checkpoints=env_paths["checkpoints"],
            logs=env_paths["logs"],
        )
        return cls(
            experiment_name=experiment_name,
            seed=42,
            paths=paths,
            hardware=hardware_config or HardwareConfig,
            curriculum=CurriculumConfig(),
            evolution=EvolutionConfig(),
            ssl=SSLConfig(),
            evaluation=EvaluationConfig(),
            training=TrainingConfig(),
        )
    def to_dict(self) -> dict:
        return {
            "experiment_name": self.experiment_name,
            "seed": self.seed,
            "hardware": self.hardware.__class__.__name__,
            "debug_mode": self.debug_mode,
        }
def create_default_config(
    experiment_name: str = "sseg_nase_default"
) -> BaseConfig:
    return BaseConfig.from_experiment_name(experiment_name)
