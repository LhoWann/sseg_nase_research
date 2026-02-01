from pathlib import Path
from typing import Optional
from configs.curriculum_config import CurriculumConfig
from configs.curriculum_config import CurriculumLevel
from data.curriculum.curriculum_dataset import CurriculumDataset
class CurriculumScheduler: 
    def __init__(
        self, 
        config: CurriculumConfig, 
        data_dir: Path,
        seed: Optional[int] = None
    ):
        self._config = config
        self._data_dir = data_dir
        self._seed = seed
        self._current_level = CurriculumLevel.BASIC
        self._datasets: dict[CurriculumLevel, CurriculumDataset] = {}

    def get_current_dataset(self) -> CurriculumDataset:
        if self._current_level not in self._datasets:
            self._datasets[self._current_level] = CurriculumDataset(
                config=self._config,
                level=self._current_level,
                data_dir=self._data_dir,
                seed=self._seed,
            )
        return self._datasets[self._current_level]
    def advance_level(self) -> bool:
        if self._current_level < CurriculumLevel.ADVERSARIAL:
            self._current_level = CurriculumLevel(self._current_level + 1)
            return True
        return False
    def reset(self) -> None:
        self._current_level = CurriculumLevel. BASIC
        self._datasets.clear()
    @property
    def current_level(self) -> CurriculumLevel:
        return self._current_level
    @property
    def is_final_level(self) -> bool:
        return self._current_level == CurriculumLevel. ADVERSARIAL
    @property
    def progress(self) -> float:
        return self._current_level / len(CurriculumLevel)
