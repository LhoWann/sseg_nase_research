from dataclasses import dataclass
import math
import numpy as np
@dataclass
class ConfidenceInterval:
    mean: float
    std: float
    margin: float
    lower: float
    upper: float
    confidence_level: float = 0.95
    def __str__(self) -> str:
        return f"{self.mean:.2f}±{self.margin:.2f}"
    @classmethod
    def from_samples(
        cls,
        samples: np.ndarray,
        confidence_level: float = 0.95,
    ) -> "ConfidenceInterval":
        mean = float(np.mean(samples))
        std = float(np.std(samples, ddof=1))
        n = len(samples)
        z_value = 1.96 if confidence_level == 0.95 else 2.576
        margin = z_value * std / math.sqrt(n)
        return cls(
            mean=mean,
            std=std,
            margin=float(margin),
            lower=mean - margin,
            upper=mean + margin,
            confidence_level=confidence_level,
        )
    @classmethod
    def from_mean_std(
        cls,
        mean: float,
        std: float,
        n: int,
        confidence_level: float = 0.95,
    ) -> "ConfidenceInterval":
        z_value = 1.96 if confidence_level == 0.95 else 2.576
        margin = z_value * std / math.sqrt(n)
        return cls(
            mean=mean,
            std=std,
            margin=float(margin),
            lower=mean - margin,
            upper=mean + margin,
            confidence_level=confidence_level,
        )
def compute_confidence_interval(
    accuracies: list[float],
    confidence_level: float = 0.95,
) -> ConfidenceInterval:
    samples = np.array(accuracies)
    return ConfidenceInterval.from_samples(samples, confidence_level)
