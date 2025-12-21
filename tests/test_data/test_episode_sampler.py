import pytest
import torch
from torch.utils.data import Dataset
from dataclasses import dataclass
from typing import Iterator
import random


@dataclass(frozen=True)
class Episode:
    support_images: torch. Tensor
    support_labels: torch. Tensor
    query_images: torch. Tensor
    query_labels: torch. Tensor


class EpisodeSampler:
    
    def __init__(
        self,
        dataset: Dataset,
        num_ways: int,
        num_shots: int,
        num_queries: int,
        num_episodes: int,
    ):
        self._dataset = dataset
        self._num_ways = num_ways
        self._num_shots = num_shots
        self._num_queries = num_queries
        self._num_episodes = num_episodes
        
        self._class_indices = self._build_class_indices()
    
    def _build_class_indices(self) -> dict: 
        class_indices:  dict = {}
        
        for idx in range(len(self._dataset)):
            _, label = self._dataset[idx]
            
            if label not in class_indices: 
                class_indices[label] = []
            class_indices[label].append(idx)
        
        return class_indices
    
    def sample_episode(self) -> Episode:
        available_classes = list(self._class_indices.keys())
        selected_classes = random.sample(available_classes, self._num_ways)
        
        support_images = []
        support_labels = []
        query_images = []
        query_labels = []
        
        for new_label, class_id in enumerate(selected_classes):
            class_samples = self._class_indices[class_id]
            
            num_required = self._num_shots + self._num_queries
            if len(class_samples) < num_required: 
                raise ValueError(
                    f"Class {class_id} has only {len(class_samples)} samples, "
                    f"but {num_required} required"
                )
            
            selected_indices = random.sample(class_samples, num_required)
            
            support_indices = selected_indices[:self._num_shots]
            query_indices = selected_indices[self._num_shots:]
            
            for idx in support_indices:
                image, _ = self._dataset[idx]
                support_images.append(image)
                support_labels.append(new_label)
            
            for idx in query_indices: 
                image, _ = self._dataset[idx]
                query_images.append(image)
                query_labels.append(new_label)
        
        return Episode(
            support_images=torch.stack(support_images),
            support_labels=torch.tensor(support_labels, dtype=torch.long),
            query_images=torch.stack(query_images),
            query_labels=torch.tensor(query_labels, dtype=torch.long),
        )
    
    def __iter__(self) -> Iterator[Episode]:
        for _ in range(self._num_episodes):
            yield self.sample_episode()
    
    def __len__(self) -> int:
        return self._num_episodes


class MockDataset(Dataset):
    
    def __init__(self, num_classes: int = 10, samples_per_class:  int = 50):
        self._num_classes = num_classes
        self._samples_per_class = samples_per_class
        self._data = []
        
        for class_idx in range(num_classes):
            for _ in range(samples_per_class):
                image = torch.randn(3, 84, 84)
                self._data.append((image, class_idx))
    
    def __len__(self) -> int:
        return len(self._data)
    
    def __getitem__(self, index:  int) -> tuple:
        return self._data[index]


class TestEpisodeSampler:
    
    @pytest.fixture
    def dataset(self) -> MockDataset:
        return MockDataset(num_classes=20, samples_per_class=50)
    
    @pytest.fixture
    def sampler(self, dataset: MockDataset) -> EpisodeSampler: 
        return EpisodeSampler(
            dataset=dataset,
            num_ways=5,
            num_shots=1,
            num_queries=15,
            num_episodes=10,
        )
    
    def test_sampler_length_matches_num_episodes(
        self, sampler:  EpisodeSampler
    ) -> None:
        assert len(sampler) == 10
    
    def test_sample_episode_returns_episode(
        self, sampler: EpisodeSampler
    ) -> None:
        episode = sampler.sample_episode()
        
        assert isinstance(episode, Episode)
    
    def test_episode_support_images_shape(
        self, sampler: EpisodeSampler
    ) -> None:
        episode = sampler.sample_episode()
        
        expected_support_size = 5 * 1
        assert episode.support_images.shape == (expected_support_size, 3, 84, 84)
    
    def test_episode_support_labels_shape(
        self, sampler: EpisodeSampler
    ) -> None:
        episode = sampler.sample_episode()
        
        expected_support_size = 5 * 1
        assert episode.support_labels.shape == (expected_support_size,)
    
    def test_episode_query_images_shape(
        self, sampler: EpisodeSampler
    ) -> None:
        episode = sampler.sample_episode()
        
        expected_query_size = 5 * 15
        assert episode.query_images.shape == (expected_query_size, 3, 84, 84)
    
    def test_episode_query_labels_shape(
        self, sampler: EpisodeSampler
    ) -> None:
        episode = sampler.sample_episode()
        
        expected_query_size = 5 * 15
        assert episode.query_labels.shape == (expected_query_size,)
    
    def test_episode_labels_are_remapped_to_zero_indexed(
        self, sampler: EpisodeSampler
    ) -> None:
        episode = sampler.sample_episode()
        
        unique_support_labels = torch.unique(episode.support_labels)
        unique_query_labels = torch.unique(episode.query_labels)
        
        assert unique_support_labels.min() == 0
        assert unique_support_labels.max() == 4
        assert unique_query_labels.min() == 0
        assert unique_query_labels.max() == 4
    
    def test_iteration_yields_correct_number_of_episodes(
        self, sampler: EpisodeSampler
    ) -> None:
        episodes = list(sampler)
        
        assert len(episodes) == 10
    
    def test_five_shot_sampler_support_size(self, dataset: MockDataset) -> None:
        sampler = EpisodeSampler(
            dataset=dataset,
            num_ways=5,
            num_shots=5,
            num_queries=15,
            num_episodes=5,
        )
        
        episode = sampler.sample_episode()
        
        expected_support_size = 5 * 5
        assert episode.support_images.shape[0] == expected_support_size
        assert episode.support_labels.shape[0] == expected_support_size
    
    def test_sampler_raises_error_if_insufficient_samples(self) -> None:
        small_dataset = MockDataset(num_classes=5, samples_per_class=10)
        
        sampler = EpisodeSampler(
            dataset=small_dataset,
            num_ways=5,
            num_shots=5,
            num_queries=10,
            num_episodes=1,
        )
        
        with pytest.raises(ValueError):
            sampler.sample_episode()
    
    def test_different_episodes_have_different_content(
        self, sampler: EpisodeSampler
    ) -> None:
        random.seed(123)
        episode1 = sampler.sample_episode()
        episode2 = sampler.sample_episode()
        
        images_equal = torch.allclose(
            episode1.support_images, episode2.support_images
        )
        
        assert not images_equal