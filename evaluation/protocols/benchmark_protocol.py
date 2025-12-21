from dataclasses import dataclass
from dataclasses import field
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn

from configs.evaluation_config import FewShotConfig
from data.benchmarks.minimagenet_dataset import MiniImageNetDataset
from data.benchmarks.episode_sampler import EpisodeSampler
from evaluation.evaluators.efficiency_evaluator import EfficiencyEvaluator
from evaluation.evaluators.efficiency_evaluator import EfficiencyMetrics
from evaluation.evaluators.fewshot_evaluator import FewShotEvaluator
from evaluation.metrics.confidence_interval import ConfidenceInterval


@dataclass
class FewShotBenchmarkResult:
    num_ways: int
    num_shots: int
    confidence_interval: ConfidenceInterval
    accuracies: list[float]


@dataclass
class BenchmarkResult:
    method_name: str
    dataset_name: str
    few_shot_results: list[FewShotBenchmarkResult]
    efficiency_metrics: EfficiencyMetrics
    
    def to_dict(self) -> dict:
        return {
            "method_name": self.method_name,
            "dataset_name": self.dataset_name,
            "few_shot_results": [
                {
                    "num_ways": fs.num_ways,
                    "num_shots": fs.num_shots,
                    "mean_accuracy": fs.confidence_interval.mean,
                    "std": fs.confidence_interval.std,
                    "margin": fs.confidence_interval.margin,
                    "ci_lower": fs.confidence_interval.lower,
                    "ci_upper": fs.confidence_interval.upper,
                }
                for fs in self.few_shot_results
            ],
            "efficiency_metrics": self.efficiency_metrics.to_dict(),
        }


class BenchmarkProtocol:
    
    def __init__(
        self,
        model: nn.Module,
        few_shot_config: FewShotConfig,
        device: str = "cuda",
    ):
        self._model = model
        self._config = few_shot_config
        self._device = device
    
    def run_benchmark(
        self,
        method_name: str,
        dataset_name: str,
        data_root: Path,
    ) -> BenchmarkResult:
        # Efficiency evaluation
        efficiency_evaluator = EfficiencyEvaluator(
            model=self._model,
            device=self._device,
        )
        efficiency_metrics = efficiency_evaluator.evaluate()
        
        # Few-shot evaluation
        try:
            dataset = MiniImageNetDataset(
                root_dir=data_root,
                split="test",
                image_size=84,
                augment=False,
            )
        except FileNotFoundError:
            # Return placeholder results if dataset not found
            return BenchmarkResult(
                method_name=method_name,
                dataset_name=dataset_name,
                few_shot_results=[],
                efficiency_metrics=efficiency_metrics,
            )
        
        few_shot_results = []
        
        for num_shots in self._config.num_shots:
            episode_sampler = EpisodeSampler(
                dataset=dataset,
                num_ways=self._config.num_ways,
                num_shots=num_shots,
                num_queries=self._config.num_queries_per_class,
                num_episodes=self._config.num_episodes,
            )
            
            evaluator = FewShotEvaluator(
                model=self._model,
                config=self._config,
                device=self._device,
            )
            
            ci, accuracies = evaluator.run_evaluation(episode_sampler)
            
            few_shot_results.append(FewShotBenchmarkResult(
                num_ways=self._config.num_ways,
                num_shots=num_shots,
                confidence_interval=ci,
                accuracies=accuracies,
            ))
        
        return BenchmarkResult(
            method_name=method_name,
            dataset_name=dataset_name,
            few_shot_results=few_shot_results,
            efficiency_metrics=efficiency_metrics,
        )
