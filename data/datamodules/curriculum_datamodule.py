from typing import Optional
from pathlib import Path
import pytorch_lightning as pl
from torch.utils.data import DataLoader
from configs.curriculum_config import CurriculumConfig
from configs.hardware_config import HardwareConfig
from data.curriculum.curriculum_scheduler import CurriculumScheduler
class CurriculumDataModule(pl.LightningDataModule):
    def __init__(
        self,
        curriculum_config: CurriculumConfig,
        hardware_config: HardwareConfig,
        data_dir: Path,
        seed: Optional[int] = None,
    ):
        super().__init__()
        self._curriculum_config = curriculum_config
        self._hardware_config = hardware_config
        self._data_dir = data_dir
        self._seed = seed
        self._scheduler = CurriculumScheduler(curriculum_config, data_dir, seed)
    def setup(self, stage: Optional[str] = None) -> None:
        pass
    def train_dataloader(self) -> DataLoader:
        dataset = self._scheduler.get_current_dataset()
        return DataLoader(
            dataset,
            batch_size=self._hardware_config.batch_size,
            shuffle=True,
            num_workers=self._hardware_config.num_workers,
            pin_memory=self._hardware_config.pin_memory,
            persistent_workers=self._hardware_config.persistent_workers,
            drop_last=True,
        )
    def advance_curriculum(self) -> bool:
        return self._scheduler.advance_level()
    @property
    def current_level(self) -> int:
        return self._scheduler.current_level
    @property
    def is_final_level(self) -> bool:
        return self._scheduler.is_final_level
