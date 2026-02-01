import torch
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
        metrics = trainer.callback_metrics
        ssl_loss = metrics.get("ssl/dino_loss")
        distill_loss = metrics.get("ssl/dino_loss")
        
        if ssl_loss is None:
            ssl_history, distill_history = pl_module.get_loss_history()
            if ssl_history:
                ssl_loss = ssl_history[-1]
                distill_loss = distill_history[-1] if distill_history else ssl_loss

        logger = pl_module.logger.experiment if hasattr(pl_module.logger, 'experiment') else None
        global_step = trainer.current_epoch

        if ssl_loss is None:
            if logger:
                logger.add_text("evolution/debug", f"[Evolution] Epoch {trainer.current_epoch}: Metrics missing, skipping check.", global_step)
            return

        if isinstance(ssl_loss, torch.Tensor):
            ssl_loss = ssl_loss.item()
        if isinstance(distill_loss, torch.Tensor):
            distill_loss = distill_loss.item()

        self._plateau_detector.update(ssl_loss, distill_loss)
        plateau_status = self._plateau_detector.check_plateau()
        pl_module.log("plateau/is_plateau", float(plateau_status.is_plateau))
        pl_module.log("plateau/ssl_delta", plateau_status.ssl_loss_delta)
        pl_module.log("plateau/distill_loss", plateau_status.current_distillation_loss)
        if logger:
            logger.add_text(
                "plateau/status",
                f"Plateau check: {plateau_status}",
                global_step
            )
            if not plateau_status.should_evolve and not plateau_status.should_advance_level:
                logger.add_text(
                    "plateau/status",
                    "Not evolving yet",
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
            msg = f"Mutasi {mutation_type.name} pada layer {target_idx}"
            logger.add_text(
                "evolution/mutation",
                msg,
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
