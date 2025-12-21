import pytorch_lightning as pl
from pytorch_lightning.callbacks import Callback

from training.lightning_modules.sseg_module import SSEGModule


class CurriculumCallback(Callback):
    
    def __init__(self):
        super().__init__()
        self._level_transition_epochs:  list[int] = []
    
    def on_train_epoch_end(self, trainer: pl.Trainer, pl_module: SSEGModule) -> None:
        datamodule = trainer.datamodule
        
        if hasattr(datamodule, "current_level"):
            current_level = datamodule.current_level
            pl_module.log("curriculum/level", float(current_level))
            
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
        return self._level_transition_epochs. copy()