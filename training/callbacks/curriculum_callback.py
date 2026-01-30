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
        
        # Forced Schedule: 10 epochs per synthetic level
        # Level 1 -> 2 at Epoch 10
        # Level 2 -> 3 at Epoch 20
        # Level 3 -> 4 at Epoch 30
        # Level 4 -> 5 at Epoch 40 (Real Data)
        scheduled_advance_epochs = {10, 20, 30, 40}
        
        should_force_advance = current_epoch in scheduled_advance_epochs
        
        if hasattr(datamodule, "current_level"):
            # Check if we are behind schedule
            # Level 1 is index 1 or enum value 1. Assuming enum value.
            # Expected level at epoch:
            # < 10: 1
            # 10-19: 2
            # 20-29: 3
            # 30-39: 4
            # >= 40: 5
            
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
                    
                    # Reset optimizer state for new level to avoid bad momentum from previous task
                    # especially when switching from Synthetic -> Real
                    for opt in trainer.optimizers:
                        opt.state.clear()
                    
                    # Reset loss history to avoid plateau detector confusion
                    pl_module.clear_loss_history()
                    
                    # Log the reset
                    if hasattr(pl_module.logger, 'experiment'):
                        pl_module.logger.experiment.add_text(
                            "curriculum/debug", 
                            f"Forced advance to Level {target_level} at Epoch {current_epoch}. Optimizer state reset.", 
                            current_epoch
                        )

            pl_module.log("curriculum/level", float(datamodule.current_level))
            
            if hasattr(datamodule, "is_final_level"):
                pl_module.log(
                    "curriculum/is_final_level", float(datamodule.is_final_level)
                )
    
    def on_train_batch_start(
        self, trainer: pl. Trainer, pl_module: SSEGModule, batch, batch_idx: int
    ) -> None:
        pass
    
    @property
    def level_transition_epochs(self) -> list[int]:
        return self._level_transition_epochs.copy()