import pytorch_lightning as pl
from pytorch_lightning.callbacks import Callback

from configs.evolution_config import EvolutionConfig
from models.evolution.architecture_tracker import ArchitectureTracker
from models.evolution.mutation_selector import MutationSelector
from training.callbacks.plateau_detector import PlateauDetector
from training.lightning_modules.sseg_module import SSEGModule


class EvolutionCallback(Callback):
    
    def __init__(self, config: EvolutionConfig):
        super().__init__()
        
        self._config = config
        
        self._plateau_detector = PlateauDetector(
            window_size=config.growth.plateau_window_size,
            plateau_threshold=config.growth.plateau_threshold,
            distillation_gap_threshold=config.growth.distillation_gap_threshold,
        )
        
        self._architecture_tracker = ArchitectureTracker()
        self._current_level = 1
    
    def on_train_epoch_end(self, trainer: pl.Trainer, pl_module: SSEGModule) -> None:
        ssl_history, distill_history = pl_module.get_loss_history()
        logger = pl_module.logger.experiment if hasattr(pl_module.logger, 'experiment') else None
        global_step = trainer.current_epoch
        if not ssl_history or not distill_history:
            if logger:
                logger.add_text("evolution/debug", f"[Evolution] Epoch {trainer.current_epoch}: Loss history kosong, tidak dicek evolusi.", global_step)
            return
        self._plateau_detector.update(ssl_history[-1], distill_history[-1])
        plateau_status = self._plateau_detector.check_plateau()
        pl_module.log("plateau/is_plateau", float(plateau_status.is_plateau))
        pl_module.log("plateau/ssl_delta", plateau_status.ssl_loss_delta)
        pl_module.log("plateau/distill_loss", plateau_status.current_distillation_loss)
        # Logging trigger evolusi dan alasan
        if logger:
            logger.add_text(
                "evolution/debug",
                f"[Evolution] Epoch {trainer.current_epoch}: is_plateau={plateau_status.is_plateau}, "
                f"ssl_loss_delta={plateau_status.ssl_loss_delta:.5f}, distill_loss={plateau_status.current_distillation_loss:.5f}, "
                f"should_evolve={plateau_status.should_evolve}, should_advance_level={plateau_status.should_advance_level}",
                global_step
            )
            if not plateau_status.should_evolve and not plateau_status.should_advance_level:
                logger.add_text(
                    "evolution/debug",
                    f"[Evolution] Tidak terjadi pertumbuhan: loss delta={plateau_status.ssl_loss_delta:.5f}, threshold={self._plateau_detector._plateau_threshold}",
                    global_step
                )
        if plateau_status.should_evolve:
            if logger:
                logger.add_text("evolution/debug", f"[Evolution] TRIGGER: Model berevolusi pada epoch {trainer.current_epoch}", global_step)
            self._trigger_evolution(trainer, pl_module, plateau_status)
        elif plateau_status.should_advance_level:
            if logger:
                logger.add_text("evolution/debug", f"[Evolution] TRIGGER: Advance curriculum level pada epoch {trainer.current_epoch}", global_step)
            self._trigger_level_advance(trainer, pl_module)
    
    def _trigger_evolution(
        self, trainer: pl.Trainer, pl_module: SSEGModule, plateau_status
    ) -> None:
        mutation_selector = MutationSelector(pl_module.backbone, self._config)
        logger = pl_module.logger.experiment if hasattr(pl_module.logger, 'experiment') else None
        global_step = trainer.current_epoch
        sensitivities = mutation_selector.compute_layer_sensitivities()
        mutation_type, target_idx = mutation_selector.select_mutation(sensitivities)
        if mutation_type.name == "NONE":
            if logger:
                logger.add_text("evolution/debug", f"[Evolution] Gagal evolusi: MutationType.NONE dipilih pada epoch {trainer.current_epoch}", global_step)
            return
        old_summary = pl_module.backbone.get_architecture_summary()
        success = pl_module.evolve_network(mutation_type.name.lower(), target_idx)
        if logger:
            logger.add_text(
                "evolution/debug",
                f"[Evolution] Percobaan grow: mutation={mutation_type.name}, target={target_idx}, sukses={success} pada epoch {trainer.current_epoch}",
                global_step
            )
        if success:
            new_summary = pl_module.backbone.get_architecture_summary()
            self._architecture_tracker.record_mutation(
                epoch=trainer.current_epoch,
                level=self._current_level,
                mutation_type=mutation_type,
                target_layer=target_idx,
                num_blocks_before=old_summary["num_blocks"],
                num_blocks_after=new_summary["num_blocks"],
                num_params_before=old_summary["total_params"],
                num_params_after=new_summary["total_params"],
                ssl_loss_before=plateau_status.current_ssl_loss,
            )
            pl_module.clear_loss_history()
            self._plateau_detector.reset()
            trainer.strategy.setup_optimizers(trainer)
        else:
            if logger:
                logger.add_text("evolution/debug", f"[Evolution] Gagal grow: Model tidak bertambah blok pada epoch {trainer.current_epoch}", global_step)
    
    def _trigger_level_advance(self, trainer: pl.Trainer, pl_module: SSEGModule) -> None:
        datamodule = trainer.datamodule
        
        if hasattr(datamodule, "advance_curriculum"):
            advanced = datamodule.advance_curriculum()
            
            if advanced: 
                self._current_level += 1
                pl_module.clear_loss_history()
                self._plateau_detector.reset()
    
    @property
    def architecture_tracker(self) -> ArchitectureTracker:
        return self._architecture_tracker