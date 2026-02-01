from typing import Literal
from typing import Optional
import torch
import torch.nn.functional as F
from torch import Tensor
class PrototypeHead: 
    def __init__(
        self,
        distance_metric: Literal["euclidean", "cosine"] = "cosine",
        normalize_features: bool = True,
        dropout: float = 0.3,
    ):
        self._distance_metric = distance_metric
        self._normalize_features = normalize_features
        self._prototypes: Optional[Tensor] = None
        self._dropout = dropout
    def fit(self, support_features: Tensor, support_labels: Tensor) -> None:
        if self._normalize_features:
            support_features = F.normalize(support_features, dim=1)
        unique_labels = torch.unique(support_labels)
        prototypes = []
        for label in unique_labels:
            mask = support_labels == label
            class_features = support_features[mask]
            prototype = class_features.mean(dim=0)
            prototypes.append(prototype)
        self._prototypes = torch.stack(prototypes)
    def predict(self, query_features: Tensor) -> Tensor:
        if self._prototypes is None:
            raise RuntimeError("Must call fit() before predict()")
        if self._normalize_features:
            query_features = F.normalize(query_features, dim=1)
            prototypes = F.normalize(self._prototypes, dim=1)
        else:
            prototypes = self._prototypes
        if self._dropout > 0 and self.training:
            query_features = F.dropout(query_features, p=self._dropout)
        if self._distance_metric == "cosine":
            similarities = torch.mm(query_features, prototypes.t())
            predictions = similarities.argmax(dim=1)
        else:
            distances = torch.cdist(query_features, prototypes)
            predictions = distances.argmin(dim=1)
        return predictions
    def predict_proba(self, query_features: Tensor) -> Tensor:
        if self._prototypes is None:
            raise RuntimeError("Must call fit() before predict_proba()")
        if self._normalize_features:
            query_features = F.normalize(query_features, dim=1)
            prototypes = F.normalize(self._prototypes, dim=1)
        else:
            prototypes = self._prototypes
        if self._distance_metric == "cosine": 
            similarities = torch.mm(query_features, prototypes.t())
            logits = similarities
        else:
            distances = torch.cdist(query_features, prototypes)
            logits = -distances
        probabilities = F.softmax(logits, dim=1)
        return probabilities
