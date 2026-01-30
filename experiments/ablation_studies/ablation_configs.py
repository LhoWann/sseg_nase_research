from dataclasses import dataclass
from dataclasses import field
from enum import Enum
from enum import auto
from typing import Optional

from configs.curriculum_config import CurriculumConfig
from configs.evolution_config import EvolutionConfig
from configs.evolution_config import GrowthConfig
from configs.evolution_config import NASEConfig
from configs.evolution_config import SeedNetworkConfig
from configs.evolution_config import FitnessConfig
from configs.ssl_config import DistillationConfig
from configs.ssl_config import SSLConfig


class AblationType(Enum):
    SEED_ONLY = auto()
    SSL_ONLY = auto()
    SSEG_ONLY = auto()
    SSEG_CURRICULUM = auto()
    SSEG_DISTILLATION = auto()
    SSEG_NASE = auto()
    FULL_PIPELINE = auto()


@dataclass
class AblationConfig: 
    name: str
    ablation_type: AblationType
    description: str
    
    enable_sseg: bool = False
    enable_nase: bool = False
    enable_curriculum: bool = False
    enable_distillation: bool = False
    
    seed_network:  Optional[SeedNetworkConfig] = None
    growth:  Optional[GrowthConfig] = None
    nase: Optional[NASEConfig] = None
    ssl: Optional[SSLConfig] = None
    curriculum: Optional[CurriculumConfig] = None
    
    def __post_init__(self) -> None:
        if self.seed_network is None: 
            self.seed_network = SeedNetworkConfig()
        
        if self.growth is None:
            self.growth = GrowthConfig()
        
        if self.nase is None:
            self.nase = NASEConfig()
        
        if self.ssl is None: 
            self.ssl = SSLConfig()
        
        if self.curriculum is None:
            self.curriculum = CurriculumConfig()


def create_ablation_configs() -> dict[AblationType, AblationConfig]:
    configs = {}
    
    configs[AblationType. SEED_ONLY] = AblationConfig(
        name="A1_seed_only",
        ablation_type=AblationType. SEED_ONLY,
        description="Seed network without any training, random initialization",
        enable_sseg=False,
        enable_nase=False,
        enable_curriculum=False,
        enable_distillation=False,
        seed_network=SeedNetworkConfig(
            initial_channels=16,
            initial_blocks=3,
        ),
    )
    
    configs[AblationType.SSL_ONLY] = AblationConfig(
        name="A2_ssl_only",
        ablation_type=AblationType.SSL_ONLY,
        description="Fixed seed network with SSL training, no evolution",
        enable_sseg=False,
        enable_nase=False,
        enable_curriculum=False,
        enable_distillation=False,
        seed_network=SeedNetworkConfig(
            initial_channels=16,
            initial_blocks=3,
        ),
        growth=GrowthConfig(
            max_blocks=3,
            plateau_window_size=999999,
        ),
    )
    
    configs[AblationType.SSEG_ONLY] = AblationConfig(
        name="A3_sseg_only",
        ablation_type=AblationType.SSEG_ONLY,
        description="SSEG evolution without curriculum, NASE, or distillation",
        enable_sseg=True,
        enable_nase=False,
        enable_curriculum=False,
        enable_distillation=False,
        seed_network=SeedNetworkConfig(
            initial_channels=16,
            initial_blocks=3,
        ),
        growth=GrowthConfig(
            max_blocks=12,
            max_channels=256,
            channel_expansion_ratio=1.5,
            plateau_window_size=10,
            plateau_threshold=1e-4,
            distillation_gap_threshold=0.0,
        ),
        nase=NASEConfig(
            sparsity_ratio=1e-6,
            pruning_interval_epochs=999999,
        ),
    )
    
    configs[AblationType.SSEG_CURRICULUM] = AblationConfig(
        name="A4_sseg_curriculum",
        ablation_type=AblationType.SSEG_CURRICULUM,
        description="SSEG evolution with curriculum guidance",
        enable_sseg=True,
        enable_nase=False,
        enable_curriculum=True,
        enable_distillation=False,
        seed_network=SeedNetworkConfig(
            initial_channels=16,
            initial_blocks=3,
        ),
        growth=GrowthConfig(
            max_blocks=12,
            max_channels=256,
            channel_expansion_ratio=1.5,
            plateau_window_size=10,
            plateau_threshold=1e-4,
            distillation_gap_threshold=0.0,
        ),
        nase=NASEConfig(
            sparsity_ratio=1e-6,
            pruning_interval_epochs=999999,
        ),
        curriculum=CurriculumConfig(
            transition_strategy="gradual",
            gradual_mixing_ratio=0.2,
        ),
    )
    
    configs[AblationType. SSEG_DISTILLATION] = AblationConfig(
        name="A5_sseg_distillation",
        ablation_type=AblationType. SSEG_DISTILLATION,
        description="SSEG evolution with curriculum and EMA distillation",
        enable_sseg=True,
        enable_nase=False,
        enable_curriculum=True,
        enable_distillation=True,
        seed_network=SeedNetworkConfig(
            initial_channels=16,
            initial_blocks=3,
        ),
        growth=GrowthConfig(
            max_blocks=12,
            max_channels=256,
            channel_expansion_ratio=1.5,
            plateau_window_size=10,
            plateau_threshold=1e-4,
            distillation_gap_threshold=0.1,
        ),
        nase=NASEConfig(
            sparsity_ratio=1e-6,
            pruning_interval_epochs=999999,
        ),
    )
    
    configs[AblationType.SSEG_NASE] = AblationConfig(
        name="A6_sseg_nase",
        ablation_type=AblationType. SSEG_NASE,
        description="SSEG evolution with NASE pruning, without curriculum",
        enable_sseg=True,
        enable_nase=True,
        enable_curriculum=False,
        enable_distillation=True,
        seed_network=SeedNetworkConfig(
            initial_channels=16,
            initial_blocks=3,
        ),
        growth=GrowthConfig(
            max_blocks=12,
            max_channels=256,
            channel_expansion_ratio=1.5,
            plateau_window_size=10,
            plateau_threshold=1e-4,
            distillation_gap_threshold=0.1,
        ),
        nase=NASEConfig(
            sparsity_ratio=0.3,
            pruning_interval_epochs=10,
            importance_metric="taylor",
            use_complementary_masks=True,
        ),
    )
    
    configs[AblationType. FULL_PIPELINE] = AblationConfig(
        name="A7_full_pipeline",
        ablation_type=AblationType.FULL_PIPELINE,
        description="Complete SSEG-NASE pipeline with all components",
        enable_sseg=True,
        enable_nase=True,
        enable_curriculum=True,
        enable_distillation=True,
        seed_network=SeedNetworkConfig(
            initial_channels=16,
            initial_blocks=3,
        ),
        growth=GrowthConfig(
            max_blocks=12,
            max_channels=256,
            channel_expansion_ratio=1.5,
            plateau_window_size=10,
            plateau_threshold=1e-4,
            distillation_gap_threshold=0.1,
        ),
        nase=NASEConfig(
            sparsity_ratio=0.3,
            pruning_interval_epochs=10,
            importance_metric="taylor",
            use_complementary_masks=True,
        ),
        curriculum=CurriculumConfig(
            transition_strategy="gradual",
            gradual_mixing_ratio=0.2,
        ),
    )
    
    return configs


def get_ablation_config(ablation_type: AblationType) -> AblationConfig:
    configs = create_ablation_configs()
    return configs[ablation_type]