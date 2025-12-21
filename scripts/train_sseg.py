import argparse
from pathlib import Path
import sys

import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping
from pytorch_lightning. callbacks import LearningRateMonitor
from pytorch_lightning. callbacks import ModelCheckpoint
from pytorch_lightning. callbacks import RichProgressBar
from pytorch_lightning.loggers import TensorBoardLogger

sys.path.insert(0, str(Path(__file__).parent.parent))

from configs.base_config import BaseConfig
from configs.base_config import PathConfig
from configs.curriculum_config import CurriculumConfig
from configs.evaluation_config import EvaluationConfig
from configs.evolution_config import EvolutionConfig
from configs. hardware_config import get_hardware_config
from configs.ssl_config import SSLConfig
from data.datamodules. curriculum_datamodule import CurriculumDataModule
from training.callbacks.architecture_logger import ArchitectureLogger
from training. callbacks.curriculum_callback import CurriculumCallback
from training. callbacks.evolution_callback import EvolutionCallback
from training.callbacks.nase_callback import NASECallback
from training.lightning_modules.sseg_module import SSEGModule
from utils.io. checkpoint_manager import CheckpointManager
from utils.io.config_loader import ConfigLoader
from utils.logging. custom_logger import get_logger
from utils.logging. custom_logger import LogLevel
from utils.reproducibility.seed_everything import seed_everything


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
        default="rtx3060",
        choices=["rtx3060", "rtx3090", "a100"],
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
    if args.config and args.config.exists():
        config_dict = ConfigLoader. load(args.config)
        experiment_name = config_dict.get("experiment_name", args.experiment_name)
        seed = config_dict. get("seed", args.seed)
    else:
        experiment_name = args. experiment_name
        seed = args.seed
    
    paths = PathConfig(
        root=args.output_dir,
        data=args.data_dir,
        outputs=args.output_dir / experiment_name,
        checkpoints=args.output_dir / experiment_name / "checkpoints",
        logs=args.output_dir / experiment_name / "logs",
        results=args.output_dir / experiment_name / "results",
    )
    
    hardware_config = get_hardware_config(args.hardware)
    
    return BaseConfig(
        experiment_name=experiment_name,
        seed=seed,
        paths=paths,
        hardware=hardware_config,
        curriculum=CurriculumConfig(),
        evolution=EvolutionConfig(),
        ssl=SSLConfig(),
        evaluation=EvaluationConfig(),
        debug_mode=args.debug,
    )


def create_callbacks(config: BaseConfig) -> list[pl.Callback]: 
    callbacks = [
        EvolutionCallback(config.evolution),
        NASECallback(config.evolution. nase),
        CurriculumCallback(),
        ArchitectureLogger(
            log_dir=config.paths.logs,
            log_interval=10,
        ),
        ModelCheckpoint(
            dirpath=config. paths.checkpoints,
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
        log_file=config.paths. logs / "training.log",
    )
    
    logger.info(f"Starting experiment: {config.experiment_name}")
    logger.info(f"Hardware profile: {args.hardware}")
    logger.info(f"Seed: {config.seed}")
    
    seed_everything(config. seed, deterministic=True)
    
    datamodule = CurriculumDataModule(
        curriculum_config=config. curriculum,
        hardware_config=config. hardware,
        seed=config.seed,
    )
    
    module = SSEGModule(config)
    
    logger.log_architecture(
        num_blocks=module.backbone.num_blocks,
        num_params=sum(p.numel() for p in module.backbone.parameters()),
        feature_dim=module. backbone.feature_dim,
    )
    
    callbacks = create_callbacks(config)
    tb_logger = create_logger_instance(config)
    
    max_epochs_total = args.max_epochs * 4
    
    trainer = pl.Trainer(
        accelerator=config.hardware.accelerator,
        devices=config.hardware.devices,
        precision=config.hardware. precision,
        max_epochs=max_epochs_total,
        accumulate_grad_batches=config.hardware.gradient_accumulation_steps,
        gradient_clip_val=config.hardware. gradient_clip_val,
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
        checkpoint_dir=config.paths. checkpoints,
        experiment_name=config. experiment_name,
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
                "num_blocks_after": m. num_blocks_after,
                "num_params_before":  m.num_params_before,
                "num_params_after": m.num_params_after,
            }
            for m in evolution_callback. architecture_tracker.mutation_history
        ]
    
    from torch.optim import AdamW
    optimizer = AdamW(module.parameters(), lr=1e-3)
    
    ssl_history, _ = module.get_loss_history()
    final_ssl_loss = ssl_history[-1] if ssl_history else 0.0
    
    checkpoint_manager.save(
        model=module. backbone,
        optimizer=optimizer,
        epoch=trainer.current_epoch,
        curriculum_level=datamodule.current_level,
        ssl_loss=final_ssl_loss,
        architecture_summary=module.backbone.get_architecture_summary(),
        mutation_history=mutation_history,
    )
    
    logger.info("Training completed successfully")
    logger.log_architecture(
        num_blocks=module.backbone.num_blocks,
        num_params=sum(p. numel() for p in module.backbone. parameters()),
        feature_dim=module. backbone.feature_dim,
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