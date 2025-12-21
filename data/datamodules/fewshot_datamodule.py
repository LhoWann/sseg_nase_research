from pathlib import Path
from typing import Optional

import pytorch_lightning as pl
from torch.utils.data import DataLoader

from configs.evaluation_config import FewShotConfig
from configs.hardware_config import HardwareConfig
from data.benchmarks.episode_sampler import EpisodeSampler
from data.benchmarks.minimagenet_dataset import MiniImageNetDataset


class FewShotDataModule(pl.LightningDataModule):
    
    def __init__(
        self,
        data_root: Path,
        few_shot_config: FewShotConfig,
        hardware_config: HardwareConfig,
        num_shots: int,
    ):
        super().__init__()
        self._data_root = data_root
        self._few_shot_config = few_shot_config
        self._hardware_config = hardware_config
        self._num_shots = num_shots
        
        self._test_dataset:  Optional[MiniImageNetDataset] = None
        self._episode_sampler: Optional[EpisodeSampler] = None
    
    def setup(self, stage: Optional[str] = None) -> None:
        if stage == "test" or stage is None:
            self._test_dataset = MiniImageNetDataset(
                root_dir=self._data_root,
                split="test",
                image_size=84,
                augment=False,
            )
            
            self._episode_sampler = EpisodeSampler(
                dataset=self._test_dataset,
                num_ways=self._few_shot_config.num_ways,
                num_shots=self._num_shots,
                num_queries=self._few_shot_config.num_queries_per_class,
                num_episodes=self._few_shot_config.num_episodes,
            )
    
    def test_dataloader(self) -> EpisodeSampler:
        return self._episode_sampler