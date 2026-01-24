from typing import Iterator
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from tqdm.auto import tqdm

from configs.evaluation_config import FewShotConfig
from data.benchmarks.episode_sampler import Episode
from data.benchmarks.episode_sampler import EpisodeSampler
from evaluation.metrics.accuracy_metrics import compute_accuracy
from evaluation.metrics.confidence_interval import ConfidenceInterval
from evaluation.metrics.confidence_interval import compute_confidence_interval


class FewShotEvaluator:
    
    def __init__(
        self,
        model: nn.Module,
        config: FewShotConfig,
        device: str = "cuda",
    ):
        self._model = model.to(device)
        self._config = config
        self._device = device
        self._model.eval()
    
    def compute_prototypes(
        self,
        support_features: Tensor,
        support_labels: Tensor,
    ) -> Tensor:
        if self._config.normalize_features:
            support_features = F.normalize(support_features, dim=1)
        
        unique_labels = torch.unique(support_labels)
        prototypes = []
        
        for label in unique_labels:
            mask = support_labels == label
            class_features = support_features[mask]
            prototype = class_features.mean(dim=0)
            prototypes.append(prototype)
        
        return torch.stack(prototypes)
    
    def classify(
        self,
        query_features: Tensor,
        prototypes: Tensor,
    ) -> Tensor:
        if self._config.normalize_features:
            query_features = F.normalize(query_features, dim=1)
            prototypes = F.normalize(prototypes, dim=1)
        
        if self._config.distance_metric == "cosine":
            similarities = torch.mm(query_features, prototypes.t())
            predictions = similarities.argmax(dim=1)
        else:
            distances = torch.cdist(query_features, prototypes)
            predictions = distances.argmin(dim=1)
        
        return predictions
    
    def evaluate_episode(self, episode: Episode) -> float:
        support_images = episode.support_images.to(self._device)
        support_labels = episode.support_labels.to(self._device)
        query_images = episode.query_images.to(self._device)
        query_labels = episode.query_labels.to(self._device)
        
        with torch.no_grad():
            support_features = self._model(support_images)
            query_features = self._model(query_images)
        
        prototypes = self.compute_prototypes(support_features, support_labels)
        predictions = self.classify(query_features, prototypes)
        
        accuracy = compute_accuracy(predictions, query_labels)
        return accuracy
    
    def run_evaluation(
        self,
        episode_sampler: EpisodeSampler,
        show_progress: bool = True,
    ) -> tuple[ConfidenceInterval, list[float]]:
        accuracies = []
        
        iterator = episode_sampler
        if show_progress:
            iterator = tqdm(episode_sampler, desc="Evaluating episodes")
        
        for episode in iterator:
            accuracy = self.evaluate_episode(episode)
            accuracies.append(accuracy)
        
        confidence_interval = compute_confidence_interval(accuracies)
        
        return confidence_interval, accuracies
