from dataclasses import dataclass
from dataclasses import field
from datetime import datetime
from pathlib import Path
from typing import Any
from typing import Optional
import json
import torch
import torch.nn as nn
from torch.optim import Optimizer
@dataclass
class CheckpointMetadata:
    experiment_name: str
    epoch: int
    curriculum_level: int
    num_blocks: int
    num_params: int
    feature_dim: int
    ssl_loss: float
    timestamp: str
    mutation_count: int = 0
    def to_dict(self) -> dict[str, Any]: 
        return {
            "experiment_name": self.experiment_name,
            "epoch": self.epoch,
            "curriculum_level": self.curriculum_level,
            "num_blocks": self.num_blocks,
            "num_params": self.num_params,
            "feature_dim": self.feature_dim,
            "ssl_loss": self.ssl_loss,
            "timestamp": self.timestamp,
            "mutation_count": self.mutation_count,
        }
    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "CheckpointMetadata":
        return cls(
            experiment_name=data["experiment_name"],
            epoch=data["epoch"],
            curriculum_level=data["curriculum_level"],
            num_blocks=data["num_blocks"],
            num_params=data["num_params"],
            feature_dim=data["feature_dim"],
            ssl_loss=data["ssl_loss"],
            timestamp=data["timestamp"],
            mutation_count=data.get("mutation_count", 0),
        )
class CheckpointManager: 
    def __init__(
        self,
        checkpoint_dir:  Path,
        experiment_name: str,
        max_checkpoints: int = 5,
    ):
        self._checkpoint_dir = checkpoint_dir
        self._experiment_name = experiment_name
        self._max_checkpoints = max_checkpoints
        self._checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self._saved_checkpoints: list[Path] = []
        self._load_existing_checkpoints()
    def _load_existing_checkpoints(self) -> None:
        pattern = f"{self._experiment_name}_epoch_*.pt"
        existing = sorted(self._checkpoint_dir.glob(pattern))
        self._saved_checkpoints = existing
    def save(
        self,
        model: nn.Module,
        optimizer:  Optimizer,
        epoch: int,
        curriculum_level: int,
        ssl_loss: float,
        architecture_summary: dict[str, Any],
        mutation_history: Optional[list[dict]] = None,
        additional_state: Optional[dict[str, Any]] = None,
    ) -> Path:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        metadata = CheckpointMetadata(
            experiment_name=self._experiment_name,
            epoch=epoch,
            curriculum_level=curriculum_level,
            num_blocks=architecture_summary["num_blocks"],
            num_params=architecture_summary["total_params"],
            feature_dim=architecture_summary["feature_dim"],
            ssl_loss=ssl_loss,
            timestamp=timestamp,
            mutation_count=len(mutation_history) if mutation_history else 0,
        )
        checkpoint = {
            "metadata": metadata.to_dict(),
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "architecture_summary": architecture_summary,
            "mutation_history": mutation_history or [],
        }
        if additional_state: 
            checkpoint["additional_state"] = additional_state
        filename = f"{self._experiment_name}_epoch_{epoch:04d}.pt"
        filepath = self._checkpoint_dir / filename
        torch.save(checkpoint, filepath)
        self._saved_checkpoints.append(filepath)
        self._cleanup_old_checkpoints()
        self._save_metadata_json(metadata, epoch)
        return filepath
    def _cleanup_old_checkpoints(self) -> None:
        while len(self._saved_checkpoints) > self._max_checkpoints:
            oldest = self._saved_checkpoints.pop(0)
            if oldest.exists():
                oldest.unlink()
    def _save_metadata_json(self, metadata: CheckpointMetadata, epoch:  int) -> None:
        metadata_file = self._checkpoint_dir / f"{self._experiment_name}_metadata.json"
        all_metadata = {}
        if metadata_file.exists():
            with open(metadata_file, "r") as f:
                all_metadata = json.load(f)
        all_metadata[f"epoch_{epoch}"] = metadata.to_dict()
        with open(metadata_file, "w") as f:
            json.dump(all_metadata, f, indent=2)
    def load(
        self,
        filepath: Optional[Path] = None,
        epoch: Optional[int] = None,
    ) -> dict[str, Any]: 
        if filepath is None:
            if epoch is not None:
                filename = f"{self._experiment_name}_epoch_{epoch:04d}.pt"
                filepath = self._checkpoint_dir / filename
            elif self._saved_checkpoints:
                filepath = self._saved_checkpoints[-1]
            else: 
                raise FileNotFoundError("No checkpoints available")
        if not filepath.exists():
            raise FileNotFoundError(f"Checkpoint not found:  {filepath}")
        checkpoint = torch.load(filepath, map_location="cpu")
        return checkpoint
    def load_model(
        self,
        model: nn.Module,
        filepath: Optional[Path] = None,
        epoch: Optional[int] = None,
    ) -> CheckpointMetadata: 
        checkpoint = self.load(filepath, epoch)
        model.load_state_dict(checkpoint["model_state_dict"])
        metadata = CheckpointMetadata.from_dict(checkpoint["metadata"])
        return metadata
    def load_full_state(
        self,
        model: nn.Module,
        optimizer: Optimizer,
        filepath: Optional[Path] = None,
        epoch: Optional[int] = None,
    ) -> tuple[CheckpointMetadata, dict[str, Any]]:
        checkpoint = self.load(filepath, epoch)
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        metadata = CheckpointMetadata.from_dict(checkpoint["metadata"])
        additional = checkpoint.get("additional_state", {})
        return metadata, additional
    def get_latest_checkpoint(self) -> Optional[Path]: 
        if self._saved_checkpoints:
            return self._saved_checkpoints[-1]
        return None
    def get_best_checkpoint(self, metric: str = "ssl_loss") -> Optional[Path]: 
        metadata_file = self._checkpoint_dir / f"{self._experiment_name}_metadata.json"
        if not metadata_file.exists():
            return self.get_latest_checkpoint()
        with open(metadata_file, "r") as f:
            all_metadata = json.load(f)
        best_epoch = None
        best_value = float("inf")
        for key, meta in all_metadata.items():
            if metric in meta and meta[metric] < best_value: 
                best_value = meta[metric]
                best_epoch = meta["epoch"]
        if best_epoch is not None:
            filename = f"{self._experiment_name}_epoch_{best_epoch: 04d}.pt"
            filepath = self._checkpoint_dir / filename
            if filepath.exists():
                return filepath
        return self.get_latest_checkpoint()
    def list_checkpoints(self) -> list[tuple[Path, CheckpointMetadata]]:
        result = []
        for filepath in self._saved_checkpoints: 
            try:
                checkpoint = torch.load(filepath, map_location="cpu")
                metadata = CheckpointMetadata.from_dict(checkpoint["metadata"])
                result.append((filepath, metadata))
            except Exception: 
                continue
        return result
