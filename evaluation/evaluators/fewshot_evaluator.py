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
        adaptation_config: Optional[dict] = None,
    ):
        self._model = model.to(device)
        self._config = config
        self._device = device
        self._adaptation_config = adaptation_config
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
        if self._adaptation_config and self._adaptation_config.get("use_ttt", False):
            ttt_steps = self._adaptation_config.get("ttt_steps", 5)
            ttt_lr = self._adaptation_config.get("ttt_lr", 1e-4)
            self._model.train()
            backbone_optimizer = torch.optim.SGD(
                self._model.parameters(), 
                lr=ttt_lr, 
                weight_decay=1e-4,
                momentum=0.9
            )
            for _ in range(ttt_steps):
                if hasattr(self._model, "get_rotation_loss"):
                    rot_loss = self._model.get_rotation_loss(support_images)
                    backbone_optimizer.zero_grad()
                    rot_loss.backward()
                    backbone_optimizer.step()
                else:
                    break
            self._model.eval()
        if self._adaptation_config and self._adaptation_config["steps"] > 0:
            with torch.no_grad():
                support_features_raw = self._model(support_images)
            feature_dim = support_features_raw.shape[1]
            num_classes = len(torch.unique(support_labels))
            classifier_type = self._adaptation_config.get("classifier_type", "linear")
            if classifier_type == "cosine":
                from models.heads.cosine_classifier import CosineClassifier
                classifier = CosineClassifier(feature_dim, num_classes, scale=10.0, learn_scale=True).to(self._device)
            else:
                 classifier = nn.Linear(feature_dim, num_classes).to(self._device)
            prototypes = self.compute_prototypes(support_features_raw, support_labels)
            if classifier_type == "cosine":
                 classifier.weight.data.copy_(F.normalize(prototypes, dim=1))
            else:
                 if self._config.normalize_features:
                      prototypes = F.normalize(prototypes, dim=1)
                 classifier.weight.data.copy_(prototypes)
                 classifier.bias.data.zero_()
            optimizer = torch.optim.SGD(
                classifier.parameters(), 
                lr=self._adaptation_config["lr"],
                momentum=0.9,
                weight_decay=1e-4
            )
            unique_labels = torch.unique(support_labels)
            label_map = {l.item(): i for i, l in enumerate(unique_labels)}
            mapped_support_labels = torch.tensor([label_map[l.item()] for l in support_labels], device=self._device)
            classifier.train()
            for _ in range(self._adaptation_config["steps"]):
                if self._config.normalize_features and classifier_type == "linear":
                     features = F.normalize(support_features_raw, dim=1)
                else:
                     features = support_features_raw
                logits = classifier(features)
                loss = F.cross_entropy(logits, mapped_support_labels)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            classifier.eval()
            with torch.no_grad():
                query_features = self._model(query_images)
                if self._config.normalize_features and classifier_type == "linear":
                    query_features = F.normalize(query_features, dim=1)
                logits = classifier(query_features)
                mapped_query_labels = torch.tensor([label_map[l.item()] for l in query_labels], device=self._device)
                predictions = logits.argmax(dim=1)
                accuracy = compute_accuracy(predictions, mapped_query_labels)
                return accuracy
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
