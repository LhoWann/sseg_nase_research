from pathlib import Path

import pytorch_lightning as pl
from pytorch_lightning.callbacks import Callback

from training.lightning_modules. sseg_module import SSEGModule


class ArchitectureLogger(Callback):
    
    def __init__(self, log_dir: Path, log_interval: int = 10):
        super().__init__()
        
        self._log_dir = log_dir
        self._log_interval = log_interval
        
        self._log_dir.mkdir(parents=True, exist_ok=True)
        
        self._architecture_log_file = self._log_dir / "architecture_evolution.txt"
    
    def on_train_epoch_end(self, trainer: pl.Trainer, pl_module: SSEGModule) -> None:
        if trainer.current_epoch % self._log_interval != 0:
            return
        
        architecture_summary = pl_module.backbone.get_architecture_summary()
        
        log_entry = self._format_log_entry(trainer. current_epoch, architecture_summary)
        
        with open(self._architecture_log_file, "a") as f:
            f.write(log_entry + "\n")
    
    def _format_log_entry(self, epoch: int, summary: dict) -> str:
        lines = [
            f"Epoch {epoch}:",
            f"  Num Blocks: {summary['num_blocks']}",
            f"  Channel Progression: {summary['channel_progression']}",
            f"  Feature Dim: {summary['feature_dim']}",
            f"  Total Params: {summary['total_params']: ,}",
            f"  Trainable Params: {summary['trainable_params']:,}",
        ]
        
        return "\n".join(lines)