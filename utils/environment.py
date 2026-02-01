import os
import sys
from pathlib import Path
from enum import Enum

class EnvironmentType(str, Enum):
    LOCAL = "local"

class EnvironmentHelper:
    @staticmethod
    def detect_environment() -> EnvironmentType:
        return EnvironmentType.LOCAL

    @staticmethod
    def get_paths(experiment_name: str) -> dict[str, Path]:
        root = Path.cwd()
        return {
            "root": root,
            "data": root / "datasets",
            "outputs": root / "outputs" / experiment_name,
            "checkpoints": root / "outputs" / experiment_name / "checkpoints",
            "logs": root / "outputs" / experiment_name / "logs"
        }