import pytorch_lightning as pl
from pytorch_lightning.callbacks import Callback
from training.lightning_modules.sseg_module import SSEGModule
class CurriculumCallback(Callback):
    def __init__(self):
        super().__init__()
        self._level_transition_epochs:  list[int] = []
    def on_train_epoch_end(self, trainer: pl.Trainer, pl_module: SSEGModule) -> None:
        datamodule = trainer.datamodule
        current_epoch = trainer.current_epoch
        scheduled_advance_epochs = {10, 20, 30, 40}
        should_force_advance = current_epoch in scheduled_advance_epochs
        if hasattr(datamodule, "current_level"):
            if current_epoch < 10:
                target_level = 1
            elif current_epoch < 20:
                target_level = 2
            elif current_epoch < 30:
                target_level = 3
            elif current_epoch < 40:
                target_level = 4
            else:
                target_level = 5
            current_level_val = int(datamodule.current_level)
            if current_level_val < target_level:
                if hasattr(datamodule, "advance_curriculum"):
                    datamodule.advance_curriculum()
                    self._level_transition_epochs.append(current_epoch)
                    for opt in trainer.optimizers:
                        opt.state.clear()
                    pl_module.clear_loss_history()
                    if hasattr(pl_module.logger, 'experiment'):
                        pl_module.logger.experiment.add_text(
                            "curriculum/update", 
                            f"Advanced to level {target_level}", 
                            current_epoch
                        
                        )
            pl_module.log("curriculum/level", float(datamodule.current_level))
            if hasattr(datamodule, "is_final_level"):
                pl_module.log(
                    "curriculum/is_final", float(datamodule.is_final_level)
                )
    def on_train_batch_start(
        self, trainer: pl. Trainer, pl_module: SSEGModule, batch, batch_idx: int
    ) -> None:
        pass
    @property
    def level_transition_epochs(self) -> list[int]:
        return self._level_transition_epochs.copy()
