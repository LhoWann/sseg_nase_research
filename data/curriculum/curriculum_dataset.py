from pathlib import Path
from typing import Optional

import torch
from torch import Tensor
from torch.utils.data import Dataset

from configs.curriculum_config import CurriculumConfig
from configs.curriculum_config import CurriculumLevel
from data.curriculum.difficulty_scorer import DifficultyScorer
from data.curriculum.synthetic_generator import GenerationConfig
from data.curriculum.synthetic_generator import SyntheticGenerator


class CurriculumDataset(Dataset):
    
    def __init__(
        self,
        config: CurriculumConfig,
        level: CurriculumLevel,
        cache_dir: Optional[Path] = None,
        transform: Optional[callable] = None,
        seed: Optional[int] = None,
    ):
        self._config = config
        self._level = level
        self._cache_dir = cache_dir
        self._transform = transform
        
        level_spec = config.get_level_spec(level)
        
        gen_config = GenerationConfig(
            image_size=config.image_size,
            seed=seed,
        )
        self._generator = SyntheticGenerator(gen_config)
        self._scorer = DifficultyScorer()
        
        self._num_samples = level_spec.num_samples
        self._samples_cache:  dict[int, Tensor] = {}
        
        if cache_dir is not None:
            cache_dir.mkdir(parents=True, exist_ok=True)
    
    def _generate_sample(self, level: CurriculumLevel) -> Tensor:
        if level == CurriculumLevel.BASIC:
            return self._generator.generate_basic_shape()
        elif level == CurriculumLevel.TEXTURE:
            return self._generator.generate_texture()
        elif level == CurriculumLevel.OBJECT:
            return self._generator.generate_complex_object()
        else:
            return self._generator.generate_adversarial()
    
    def __len__(self) -> int:
        return self._num_samples
    
    def __getitem__(self, index: int) -> Tensor:
        if index >= self._num_samples:
            raise IndexError(f"Index {index} out of range for dataset size {self._num_samples}")
        
        if index in self._samples_cache:
            image = self._samples_cache[index]
        else:
            torch.manual_seed(index)
            image = self._generate_sample(self._level)
            
            if len(self._samples_cache) < 1000:
                self._samples_cache[index] = image
        
        if self._transform is not None:
            image = self._transform(image)
        
        return image
    
    def get_difficulty_score(self, index: int) -> float:
        image = self[index]
        components = self._scorer.score(image)
        return components.aggregate(self._config.difficulty_weights)