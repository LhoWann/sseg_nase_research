import pytorch_lightning as pl
from pytorch_lightning.callbacks import Callback
from configs.evolution_config import NASEConfig
from models.nase.sparse_router import SparseRouter
from training.lightning_modules.sseg_module import SSEGModule
class NASECallback(Callback):
    def __init__(self, config: NASEConfig):
        super().__init__()
        self._config = config
        self._sparse_router = SparseRouter(config)
    def on_train_epoch_end(self, trainer: pl.Trainer, pl_module: SSEGModule) -> None:
        if (trainer.current_epoch + 1) % self._config.pruning_interval_epochs != 0:
            return
        self._sparse_router.update_masks(pl_module.backbone)
        statistics = self._sparse_router.get_statistics()
        if statistics:
            pl_module.log("nase/sparsity_ratio", statistics.sparsity_ratio)
            pl_module.log(
                "nase/active_positive",
                float(statistics.active_positive_connections),
            )
            pl_module.log(
                "nase/active_negative",
                float(statistics.active_negative_connections),
            )
    @property
    def sparse_router(self) -> SparseRouter:
        return self._sparse_router
