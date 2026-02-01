import os
import sys
from pathlib import Path
from enum import Enum

class EnvironmentType(str, Enum):
    LOCAL = "local"
    KAGGLE = "kaggle"
    COLAB = "colab"

class EnvironmentHelper:
    @staticmethod
    def detect_environment() -> EnvironmentType:
        if "google.colab" in sys.modules:
            return EnvironmentType.COLAB
        if os.path.exists("/kaggle"):
            return EnvironmentType.KAGGLE
        return EnvironmentType.LOCAL

    @staticmethod
    def get_paths(experiment_name: str) -> dict[str, Path]:
        env = EnvironmentHelper.detect_environment()
        
        if env == EnvironmentType.KAGGLE:
            root = Path("/kaggle/working")
            data_dir = Path("/kaggle/input")
            
            output_dir = root / "outputs" / experiment_name
            return {
                "root": root,
                "data": data_dir,
                "outputs": output_dir,
                "checkpoints": output_dir / "checkpoints",
                "logs": output_dir / "logs"
            }
            
        elif env == EnvironmentType.COLAB:
            root = Path("/content")
            data_dir = root / "data"
            output_dir = root / "outputs" / experiment_name
            
            drive_path = Path("/content/drive/MyDrive")
            if drive_path.exists():
                output_dir = drive_path / "sseg_research" / "outputs" / experiment_name
                
            return {
                "root": root,
                "data": data_dir,
                "outputs": output_dir,
                "checkpoints": output_dir / "checkpoints",
                "logs": output_dir / "logs"
            }
            
        else:
            root = Path.cwd()
            return {
                "root": root,
                "data": root / "datasets",
                "outputs": root / "outputs" / experiment_name,
                "checkpoints": root / "outputs" / experiment_name / "checkpoints",
                "logs": root / "outputs" / experiment_name / "logs"
            }
