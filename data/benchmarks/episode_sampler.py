import random
from dataclasses import dataclass
from typing import Iterator
import torch
from torch import Tensor
from torch.utils.data import Dataset
@dataclass(frozen=True)
class Episode:
    support_images:  Tensor
    support_labels:  Tensor
    query_images:  Tensor
    query_labels:  Tensor
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
    def _build_class_indices(self) -> dict[int, list[int]]:
        class_indices:  dict[int, list[int]] = {}
        for idx in range(len(self._dataset)):
            _, label = self._dataset[idx]
            if label not in class_indices:
                class_indices[label] = []
            class_indices[label].append(idx)
        return class_indices
    def sample_episode(self) -> Episode:
        available_classes = list(self._class_indices.keys())
        if len(available_classes) < self._num_ways:
            raise ValueError(
            )
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
                )
            selected_indices = random.sample(class_samples, num_required)
            support_indices = selected_indices[: self._num_shots]
            query_indices = selected_indices[self._num_shots :]
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
