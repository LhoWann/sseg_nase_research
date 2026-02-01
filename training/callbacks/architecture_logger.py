from pathlib import Path
import pytorch_lightning as pl
from pytorch_lightning.callbacks import Callback
from training.lightning_modules.sseg_module import SSEGModule
class ArchitectureLogger(Callback):
    def __init__(self, log_dir: Path, log_interval: int = 10):
        super().__init__()
        self._log_dir = log_dir
        self._log_interval = log_interval
        self._log_dir.mkdir(parents=True, exist_ok=True)
        self._architecture_log_file = self._log_dir / "architecture_evolution.txt"
        self._last_logged_epoch = -1
    def on_train_start(self, trainer: pl.Trainer, pl_module: SSEGModule) -> None:
        architecture_summary = pl_module.backbone.get_architecture_summary()
        log_entry = self._format_log_entry(0, architecture_summary)
        with open(self._architecture_log_file, "w") as f:
            f.write(log_entry + "\n")
        self._last_logged_epoch = 0
    def on_train_epoch_end(self, trainer: pl.Trainer, pl_module: SSEGModule) -> None:
        if trainer.current_epoch % self._log_interval != 0:
            return
        if trainer.current_epoch == self._last_logged_epoch:
            return
        architecture_summary = pl_module.backbone.get_architecture_summary()
        log_entry = self._format_log_entry(trainer.current_epoch, architecture_summary)
        with open(self._architecture_log_file, "a") as f:
            f.write(log_entry + "\n")
        self._last_logged_epoch = trainer.current_epoch
    def on_train_end(self, trainer: pl.Trainer, pl_module: SSEGModule) -> None:
        if trainer.current_epoch == self._last_logged_epoch:
            return
        architecture_summary = pl_module.backbone.get_architecture_summary()
        log_entry = self._format_log_entry(trainer.current_epoch, architecture_summary)
        with open(self._architecture_log_file, "a") as f:
            f.write(log_entry + "\n")
    def _format_log_entry(self, epoch: int, summary: dict) -> str:
        lines = [
            f"Epoch: {epoch}",
            f"Blocks: {summary.get('num_blocks', '?')}",
            f"Channels: {summary.get('num_channels', '?')}",
            f"Params: {summary.get('total_params', '?') / 1e6:.2f}M",
            f"FLOPS: {summary.get('flops', '?')}",
            "-" * 40,
        ]
        return "\n".join(lines)

