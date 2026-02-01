from dataclasses import dataclass
from typing import Iterator
import torch
from torch import Tensor
from data.benchmarks.episode_sampler import Episode
from data.benchmarks.episode_sampler import EpisodeSampler
@dataclass
class EpisodeEvaluationResult:
    episode_id: int
    accuracy: float
    num_correct: int
    num_total: int
    class_accuracies: dict[int, float]
class EpisodeProtocol:
    def __init__(
        self,
        num_ways: int,
        num_shots: int,
        num_queries: int,
        num_episodes: int,
    ):
        self._num_ways = num_ways
        self._num_shots = num_shots
        self._num_queries = num_queries
        self._num_episodes = num_episodes
    def validate_episode(self, episode: Episode) -> bool:
        expected_support_size = self._num_ways * self._num_shots
        expected_query_size = self._num_ways * self._num_queries
        support_valid = episode.support_images.size(0) == expected_support_size
        query_valid = episode.query_images.size(0) == expected_query_size
        unique_support_labels = torch.unique(episode.support_labels)
        unique_query_labels = torch.unique(episode.query_labels)
        labels_valid = (
            len(unique_support_labels) == self._num_ways and
            len(unique_query_labels) == self._num_ways
        )
        return support_valid and query_valid and labels_valid
    def compute_episode_statistics(
        self,
        predictions: Tensor,
        targets: Tensor,
    ) -> EpisodeEvaluationResult:
        num_correct = (predictions == targets).sum().item()
        num_total = targets.numel()
        accuracy = num_correct / num_total * 100.0
        class_accuracies = {}
        for class_idx in range(self._num_ways):
            mask = targets == class_idx
            if mask.sum() > 0:
                class_correct = (predictions[mask] == targets[mask]).sum().item()
                class_total = mask.sum().item()
                class_accuracies[class_idx] = class_correct / class_total * 100.0
        return EpisodeEvaluationResult(
            episode_id=0,
            accuracy=accuracy,
            num_correct=num_correct,
            num_total=num_total,
            class_accuracies=class_accuracies,
        )
    @property
    def num_ways(self) -> int:
        return self._num_ways
    @property
    def num_shots(self) -> int:
        return self._num_shots
    @property
    def num_queries(self) -> int:
        return self._num_queries
    @property
    def num_episodes(self) -> int:
        return self._num_episodes
