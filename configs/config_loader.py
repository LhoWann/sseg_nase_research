from pathlib import Path
from typing import Any

import yaml

from configs.base_config import BaseConfig
from configs.base_config import PathConfig
from configs.curriculum_config import CurriculumConfig
from configs.evaluation_config import EvaluationConfig
from configs.evaluation_config import FewShotConfig
from configs.evaluation_config import MetricsConfig
from configs.evolution_config import EvolutionConfig
from configs.evolution_config import FitnessConfig
from configs.evolution_config import GrowthConfig
from configs.evolution_config import NASEConfig
from configs.evolution_config import SeedNetworkConfig
from configs.hardware_config import get_hardware_config
from configs.ssl_config import AugmentationConfig
from configs.ssl_config import ContrastiveLossConfig
from configs.ssl_config import DistillationConfig
from configs.ssl_config import ProjectionConfig
from configs.ssl_config import SSLConfig


def load_yaml(path: Path) -> dict[str, Any]:
    with open(path, "r") as file:
        return yaml.safe_load(file)


def build_config_from_yaml(yaml_path: Path) -> BaseConfig:
    config_dict = load_yaml(yaml_path)
    
    root_dir = Path("./")
    experiment_name = config_dict["experiment_name"]
    
    paths = PathConfig(
        root=root_dir,
        data=root_dir / "datasets",
        outputs=root_dir / "outputs" / experiment_name,
        checkpoints=root_dir / "outputs" / experiment_name / "checkpoints",
        logs=root_dir / "outputs" / experiment_name / "logs",
        results=root_dir / "outputs" / experiment_name / "results",
    )
    
    hardware = get_hardware_config(config_dict["hardware"]["name"])
    
    curriculum = CurriculumConfig(
        image_size=config_dict["curriculum"]["image_size"],
        transition_strategy=config_dict["curriculum"]["transition_strategy"],
    )
    
    seed_net_cfg = config_dict["evolution"]["seed_network"]
    seed_network = SeedNetworkConfig(
        architecture=seed_net_cfg["architecture"],
        initial_channels=seed_net_cfg["initial_channels"],
        initial_blocks=seed_net_cfg["initial_blocks"],
    )
    
    growth_cfg = config_dict["evolution"]["growth"]
    growth = GrowthConfig(
        max_blocks=growth_cfg["max_blocks"],
        max_channels=growth_cfg["max_channels"],
        channel_expansion_ratio=growth_cfg["channel_expansion_ratio"],
        plateau_window_size=growth_cfg["plateau_window_size"],
        plateau_threshold=growth_cfg["plateau_threshold"],
        distillation_gap_threshold=growth_cfg["distillation_gap_threshold"],
    )
    
    nase_cfg = config_dict["evolution"]["nase"]
    nase = NASEConfig(
        sparsity_ratio=nase_cfg["sparsity_ratio"],
        pruning_interval_epochs=nase_cfg["pruning_interval_epochs"],
        importance_metric=nase_cfg["importance_metric"],
        min_channels_per_layer=nase_cfg["min_channels_per_layer"],
    )
    
    fitness_cfg = config_dict["evolution"]["fitness"]
    fitness = FitnessConfig(
        alpha_complexity_penalty=fitness_cfg["alpha_complexity_penalty"],
        target_flops_giga=fitness_cfg.get("target_flops_giga", 1.0),
        target_params_million=fitness_cfg.get("target_params_million", 1.0),
    )
    
    evolution = EvolutionConfig(
        seed_network=seed_network,
        growth=growth,
        nase=nase,
        fitness=fitness,
    )
    
    aug_cfg = config_dict["ssl"]["augmentation"]
    augmentation = AugmentationConfig(
        crop_scale_min=aug_cfg["crop_scale_min"],
        crop_scale_max=aug_cfg["crop_scale_max"],
        color_jitter_strength=aug_cfg["color_jitter_strength"],
        gaussian_blur_prob=aug_cfg["gaussian_blur_prob"],
    )
    
    cont_cfg = config_dict["ssl"]["contrastive"]
    contrastive = ContrastiveLossConfig(
        temperature=cont_cfg["temperature"],
        loss_type=cont_cfg.get("loss_type", "ntxent"),
    )
    
    dist_cfg = config_dict["ssl"]["distillation"]
    distillation = DistillationConfig(
        ema_decay=dist_cfg["ema_decay"],
        distillation_weight=dist_cfg["distillation_weight"],
        distillation_loss=dist_cfg.get("distillation_loss", "mse"),
    )
    
    proj_cfg = config_dict["ssl"]["projection"]
    projection = ProjectionConfig(
        hidden_dim=proj_cfg["hidden_dim"],
        output_dim=proj_cfg["output_dim"],
        num_layers=proj_cfg.get("num_layers", 2),
    )
    
    ssl = SSLConfig(
        augmentation=augmentation,
        contrastive=contrastive,
        distillation=distillation,
        projection=projection,
    )
    
    fs_cfg = config_dict["evaluation"]["few_shot"]
    few_shot = FewShotConfig(
        num_ways=fs_cfg["num_ways"],
        num_shots=tuple(fs_cfg["num_shots"]),
        num_queries_per_class=fs_cfg["num_queries_per_class"],
        num_episodes=fs_cfg["num_episodes"],
        distance_metric=fs_cfg.get("distance_metric", "cosine"),
    )
    
    metrics = MetricsConfig()
    
    evaluation = EvaluationConfig(
        few_shot=few_shot,
        metrics=metrics,
    )
    
    return BaseConfig(
        experiment_name=experiment_name,
        seed=config_dict["seed"],
        paths=paths,
        hardware=hardware,
        curriculum=curriculum,
        evolution=evolution,
        ssl=ssl,
        evaluation=evaluation,
        debug_mode=config_dict.get("debug_mode", False),
    )