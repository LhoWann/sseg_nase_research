import argparse
from pathlib import Path
import sys
import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping
from pytorch_lightning.callbacks import LearningRateMonitor
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.callbacks import RichProgressBar
from pytorch_lightning.loggers import TensorBoardLogger
sys.path.insert(0, str(Path(__file__).parent.parent))
from configs.base_config import BaseConfig
from configs.base_config import PathConfig
from configs.curriculum_config import CurriculumConfig
from configs.evaluation_config import EvaluationConfig
from configs.evolution_config import EvolutionConfig
from configs.hardware_config import get_hardware_config
from configs.ssl_config import SSLConfig
from data.datamodules.curriculum_datamodule import CurriculumDataModule
from training.callbacks.architecture_logger import ArchitectureLogger
from training.callbacks.curriculum_callback import CurriculumCallback
from training.callbacks.evolution_callback import EvolutionCallback
from training.callbacks.nase_callback import NASECallback
from training.lightning_modules.sseg_module import SSEGModule
from utils.io.checkpoint_manager import CheckpointManager
from utils.io.config_loader import ConfigLoader
from utils.logging.custom_logger import get_logger
from utils.logging.custom_logger import LogLevel
from utils.reproducibility.seed_everything import seed_everything
import warnings
warnings.filterwarnings("ignore", category=UserWarning)
def parse_arguments() -> argparse. Namespace:
    parser = argparse.ArgumentParser(
        description="Train SSEG-NASE pipeline for few-shot learning"
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help="Path to experiment config YAML file",
    )
    parser.add_argument(
        "--experiment-name",
        type=str,
        default="sseg_nase_experiment",
        help="Name of the experiment",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("./outputs"),
        help="Output directory for checkpoints and logs",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("./datasets"),
        help="Directory containing datasets",
    )
    parser.add_argument(
        "--hardware",
        type=str,
        default="default_gpu",
        choices=["default_gpu", "cpu"],
        help="Hardware profile for optimization",
    )
    parser.add_argument(
        "--max-epochs",
        type=int,
        default=100,
        help="Maximum epochs per curriculum level",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility",
    )
    parser.add_argument(
        "--resume",
        type=Path,
        default=None,
        help="Path to checkpoint to resume from",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug mode with reduced data",
    )
    return parser.parse_args()
def create_config(args:  argparse.Namespace) -> BaseConfig:
    config_dict = None
    if args.config and args.config.exists():
        config_dict = ConfigLoader.load(args.config)
        experiment_name = config_dict.get("experiment_name", args.experiment_name)
        seed = config_dict.get("seed", args.seed)
    else:
        experiment_name = args.experiment_name
        seed = args.seed
    paths = PathConfig(
        root=args.output_dir,
        data=args.data_dir,
        outputs=args.output_dir / experiment_name,
        checkpoints=args.output_dir / experiment_name / "checkpoints",
        logs=args.output_dir / experiment_name / "logs",
    )
    hardware_config = get_hardware_config(args.hardware)
    def get_nested(cfg, key, default=None):
        if not cfg:
            return default
        parts = key.split('.')
        cur = cfg
        for p in parts:
            if isinstance(cur, dict) and p in cur:
                cur = cur[p]
            else:
                return default
        return cur
    curriculum = CurriculumConfig(
        image_size=get_nested(config_dict, "curriculum.image_size", 84),
        transition_strategy=get_nested(config_dict, "curriculum.transition_strategy", "gradual"),
        gradual_mixing_ratio=get_nested(config_dict, "curriculum.gradual_mixing_ratio", 0.2),
    )
    from configs.evolution_config import SeedNetworkConfig, GrowthConfig, NASEConfig, FitnessConfig, EvolutionConfig
    seed_network = SeedNetworkConfig(
        architecture=get_nested(config_dict, "evolution.seed_network.architecture", "cnn"),
        initial_channels=get_nested(config_dict, "evolution.seed_network.initial_channels", 16),
        initial_blocks=get_nested(config_dict, "evolution.seed_network.initial_blocks", 3),
        kernel_size=get_nested(config_dict, "evolution.seed_network.kernel_size", 3),
        activation=get_nested(config_dict, "evolution.seed_network.activation", "relu"),
        use_batch_norm=get_nested(config_dict, "evolution.seed_network.use_batch_norm", True),
        use_pooling=get_nested(config_dict, "evolution.seed_network.use_pooling", True),
    )
    growth = GrowthConfig(
        max_blocks=get_nested(config_dict, "evolution.growth.max_blocks", 12),
        max_channels=get_nested(config_dict, "evolution.growth.max_channels", 256),
        channel_expansion_ratio=get_nested(config_dict, "evolution.growth.channel_expansion_ratio", 1.5),
        plateau_window_size=get_nested(config_dict, "evolution.growth.plateau_window_size", 10),
        plateau_threshold=get_nested(config_dict, "evolution.growth.plateau_threshold", 1e-4),
        distillation_gap_threshold=get_nested(config_dict, "evolution.growth.distillation_gap_threshold", 0.1),
        sensitivity_method=get_nested(config_dict, "evolution.growth.sensitivity_method", "taylor"),
    )
    nase = NASEConfig(
        sparsity_ratio=get_nested(config_dict, "evolution.nase.sparsity_ratio", 0.3),
        pruning_interval_epochs=get_nested(config_dict, "evolution.nase.pruning_interval_epochs", 10),
        importance_metric=get_nested(config_dict, "evolution.nase.importance_metric", "taylor"),
        min_channels_per_layer=get_nested(config_dict, "evolution.nase.min_channels_per_layer", 8),
        use_complementary_masks=get_nested(config_dict, "evolution.nase.use_complementary_masks", True),
        negative_scale=get_nested(config_dict, "evolution.nase.negative_scale", 0.1),
    )
    fitness = FitnessConfig(
        alpha_complexity_penalty=get_nested(config_dict, "evolution.fitness.alpha_complexity_penalty", 0.1),
        target_flops_giga=get_nested(config_dict, "evolution.fitness.target_flops_giga", 1.0),
        target_params_million=get_nested(config_dict, "evolution.fitness.target_params_million", 1.0),
    )
    evolution = EvolutionConfig(
        seed_network=seed_network,
        growth=growth,
        nase=nase,
        fitness=fitness,
    )
    from configs.ssl_config import AugmentationConfig, ContrastiveLossConfig, DistillationConfig, ProjectionConfig, SSLConfig, RotationLossConfig
    augmentation = AugmentationConfig(
        crop_scale_min=get_nested(config_dict, "ssl.augmentation.crop_scale_min", 0.2),
        crop_scale_max=get_nested(config_dict, "ssl.augmentation.crop_scale_max", 1.0),
        horizontal_flip_prob=get_nested(config_dict, "ssl.augmentation.horizontal_flip_prob", 0.5),
        color_jitter_strength=get_nested(config_dict, "ssl.augmentation.color_jitter_strength", 0.4),
        grayscale_prob=get_nested(config_dict, "ssl.augmentation.grayscale_prob", 0.2),
        gaussian_blur_prob=get_nested(config_dict, "ssl.augmentation.gaussian_blur_prob", 0.5),
        gaussian_blur_kernel_size=get_nested(config_dict, "ssl.augmentation.gaussian_blur_kernel_size", 9),
    )
    contrastive = ContrastiveLossConfig(
        temperature=get_nested(config_dict, "ssl.contrastive.temperature", 0.5),
        loss_type=get_nested(config_dict, "ssl.contrastive.loss_type", "ntxent"),
        normalize_features=get_nested(config_dict, "ssl.contrastive.normalize_features", True),
    )
    distillation = DistillationConfig(
        ema_decay=get_nested(config_dict, "ssl.distillation.ema_decay", 0.999),
        distillation_weight=get_nested(config_dict, "ssl.distillation.distillation_weight", 0.5),
        distillation_loss=get_nested(config_dict, "ssl.distillation.distillation_loss", "mse"),
        update_interval=get_nested(config_dict, "ssl.distillation.update_interval", 1),
    )
    projection = ProjectionConfig(
        hidden_dim=get_nested(config_dict, "ssl.projection.hidden_dim", 256),
        output_dim=get_nested(config_dict, "ssl.projection.output_dim", 128),
        num_layers=get_nested(config_dict, "ssl.projection.num_layers", 2),
        use_batch_norm=get_nested(config_dict, "ssl.projection.use_batch_norm", True),
    )
    rotation = RotationLossConfig(
        enabled=get_nested(config_dict, "ssl.rotation.enabled", False),
        weight=get_nested(config_dict, "ssl.rotation.weight", 0.5),
    )
    ssl = SSLConfig(
        augmentation=augmentation,
        contrastive=contrastive,
        distillation=distillation,
        projection=projection,
        rotation=rotation,
    )
    evaluation = EvaluationConfig()
    return BaseConfig(
        experiment_name=experiment_name,
        seed=seed,
        paths=paths,
        hardware=hardware_config,
        curriculum=curriculum,
        evolution=evolution,
        ssl=ssl,
        evaluation=evaluation,
        debug_mode=args.debug,
    )
def create_callbacks(config: BaseConfig) -> list[pl.Callback]: 
    callbacks = [
        EvolutionCallback(config.evolution),
        NASECallback(config.evolution.nase),
        CurriculumCallback(),
        ArchitectureLogger(
            log_dir=config.paths.logs,
            log_interval=10,
        ),
        ModelCheckpoint(
            dirpath=config.paths.checkpoints,
            filename="sseg-{epoch:03d}-{train/total_loss:.4f}",
            save_top_k=3,
            monitor="train/total_loss",
            mode="min",
        ),
        LearningRateMonitor(logging_interval="epoch"),
        RichProgressBar(),
        EarlyStopping(
            monitor="train/total_loss",
            patience=20,
            mode="min",
            verbose=True,
        ),
    ]
    return callbacks
def create_logger_instance(config: BaseConfig) -> TensorBoardLogger:
    return TensorBoardLogger(
        save_dir=config.paths.logs,
        name=config.experiment_name,
    )
def train(config: BaseConfig, args: argparse. Namespace) -> SSEGModule: 
    logger = get_logger(
        name="train_sseg",
        level=LogLevel.DEBUG if args.debug else LogLevel.INFO,
        log_file=config.paths.logs / "training.log",
    )
    logger.info(f"Starting experiment: {config.experiment_name}")
    logger.info(f"Hardware profile: {args.hardware}")
    logger.info(f"Seed: {config.seed}")
    logger.info(f"Evolution config: initial_channels={config.evolution.seed_network.initial_channels}, initial_blocks={config.evolution.seed_network.initial_blocks}, max_blocks={config.evolution.growth.max_blocks}, max_channels={config.evolution.growth.max_channels}, channel_expansion_ratio={config.evolution.growth.channel_expansion_ratio}, plateau_threshold={config.evolution.growth.plateau_threshold}, alpha_complexity_penalty={config.evolution.fitness.alpha_complexity_penalty}")
    logger.info(f"SSL config: distillation_weight={config.ssl.distillation.distillation_weight}, ema_decay={config.ssl.distillation.ema_decay}, projection_dim={config.ssl.projection.output_dim}")
    seed_everything(config.seed, deterministic=True)
    datamodule = CurriculumDataModule(
        curriculum_config=config.curriculum,
        hardware_config=config.hardware,
        data_dir=config.paths.data,
        seed=config.seed,
    )
    module = SSEGModule(config)
    logger.log_architecture(
        num_blocks=module.backbone.num_blocks,
        num_params=sum(p.numel() for p in module.backbone.parameters()),
        feature_dim=module.backbone.feature_dim,
    )
    callbacks = create_callbacks(config)
    tb_logger = create_logger_instance(config)
    callbacks = create_callbacks(config)
    tb_logger = create_logger_instance(config)
    
    config_max_epochs = None
    if args.config and args.config.exists():
        cfg_dict = ConfigLoader.load(args.config)
        config_max_epochs = cfg_dict.get("training", {}).get("max_epochs")
    
    max_epochs_total = config_max_epochs if config_max_epochs is not None else args.max_epochs
    trainer = pl.Trainer(
        accelerator=config.hardware.accelerator,
        devices=config.hardware.devices,
        precision=config.hardware.precision,
        max_epochs=max_epochs_total,
        accumulate_grad_batches=config.hardware.gradient_accumulation_steps,
        gradient_clip_val=config.hardware.gradient_clip_val,
        callbacks=callbacks,
        logger=tb_logger,
        enable_progress_bar=True,
        log_every_n_steps=10,
        fast_dev_run=5 if args.debug else False,
    )
    if args.resume and args.resume.exists():
        logger.info(f"Resuming from checkpoint: {args.resume}")
        trainer.fit(module, datamodule, ckpt_path=str(args.resume))
    else: 
        trainer.fit(module, datamodule)
    final_checkpoint_path = config.paths.checkpoints / "final_model.pt"
    checkpoint_manager = CheckpointManager(
        checkpoint_dir=config.paths.checkpoints,
        experiment_name=config.experiment_name,
    )
    evolution_callback = None
    for callback in callbacks:
        if isinstance(callback, EvolutionCallback):
            evolution_callback = callback
            break
    mutation_history = []
    if evolution_callback: 
        mutation_history = [
            {
                "epoch": m.epoch,
                "level": m.level,
                "mutation_type": m.mutation_type,
                "target_layer": m.target_layer,
                "num_blocks_before": m.num_blocks_before,
                "num_blocks_after": m.num_blocks_after,
                "num_params_before":  m.num_params_before,
                "num_params_after": m.num_params_after,
            }
            for m in evolution_callback.architecture_tracker.mutation_history
        ]
    from torch.optim import AdamW
    optimizer = AdamW(module.parameters(), lr=1e-3)
    ssl_history, _ = module.get_loss_history()
    final_ssl_loss = ssl_history[-1] if ssl_history else 0.0
    last_ckpt_path = checkpoint_manager.save(
        model=module.backbone,
        optimizer=optimizer,
        epoch=trainer.current_epoch,
        curriculum_level=datamodule.current_level,
        ssl_loss=final_ssl_loss,
        architecture_summary=module.backbone.get_architecture_summary(),
        mutation_history=mutation_history,
    )
    import shutil
    final_model_path = config.paths.checkpoints / "final_model.pt"
    shutil.copyfile(last_ckpt_path, final_model_path)
    logger.info("Training completed successfully")
    logger.log_architecture(
        num_blocks=module.backbone.num_blocks,
        num_params=sum(p.numel() for p in module.backbone.parameters()),
        feature_dim=module.backbone.feature_dim,
    )
    return module
def main() -> None:
    args = parse_arguments()
    config = create_config(args)
    trained_module = train(config, args)
    print(f"\nTraining completed:  {config.experiment_name}")
    print(f"Checkpoints saved to: {config.paths.checkpoints}")
    print(f"Logs saved to: {config.paths.logs}")
if __name__ == "__main__": 
    main()
