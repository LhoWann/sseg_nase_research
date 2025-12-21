from dataclasses import dataclass
from dataclasses import field
from pathlib import Path
from typing import Optional

import json

from models.evolution.evolution_operators import MutationType


@dataclass
class MutationRecord:
    epoch: int
    level: int
    mutation_type:  str
    target_layer: Optional[int]
    num_blocks_before: int
    num_blocks_after:  int
    num_params_before: int
    num_params_after: int
    ssl_loss_before: float
    ssl_loss_after:  Optional[float] = None


@dataclass
class ArchitectureTracker:
    mutation_history: list[MutationRecord] = field(default_factory=list)
    architecture_snapshots: list[dict] = field(default_factory=list)
    
    def record_mutation(
        self,
        epoch: int,
        level: int,
        mutation_type: MutationType,
        target_layer: Optional[int],
        num_blocks_before: int,
        num_blocks_after: int,
        num_params_before: int,
        num_params_after: int,
        ssl_loss_before: float,
    ) -> None:
        record = MutationRecord(
            epoch=epoch,
            level=level,
            mutation_type=mutation_type.name,
            target_layer=target_layer,
            num_blocks_before=num_blocks_before,
            num_blocks_after=num_blocks_after,
            num_params_before=num_params_before,
            num_params_after=num_params_after,
            ssl_loss_before=ssl_loss_before,
        )
        
        self.mutation_history.append(record)
    
    def record_architecture(self, epoch: int, architecture_summary: dict) -> None:
        snapshot = {
            "epoch": epoch,
            **architecture_summary,
        }
        
        self.architecture_snapshots.append(snapshot)
    
    def save(self, filepath: Path) -> None:
        data = {
            "mutation_history": [
                {
                    "epoch":  m.epoch,
                    "level": m.level,
                    "mutation_type": m.mutation_type,
                    "target_layer": m.target_layer,
                    "num_blocks_before": m.num_blocks_before,
                    "num_blocks_after": m.num_blocks_after,
                    "num_params_before": m.num_params_before,
                    "num_params_after": m.num_params_after,
                    "ssl_loss_before": m.ssl_loss_before,
                    "ssl_loss_after": m.ssl_loss_after,
                }
                for m in self.mutation_history
            ],
            "architecture_snapshots": self.architecture_snapshots,
        }
        
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        with open(filepath, "w") as f:
            json. dump(data, f, indent=2)
    
    @classmethod
    def load(cls, filepath: Path) -> "ArchitectureTracker":
        with open(filepath, "r") as f:
            data = json.load(f)
        
        mutation_history = [
            MutationRecord(**record) for record in data["mutation_history"]
        ]
        
        return cls(
            mutation_history=mutation_history,
            architecture_snapshots=data["architecture_snapshots"],
        )