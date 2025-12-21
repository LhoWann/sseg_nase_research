from dataclasses import dataclass
from dataclasses import field
from pathlib import Path
from typing import Optional
import json
import time

import pytorch_lightning as pl
import torch

from configs.base_config import BaseConfig
from configs.base_config import PathConfig
from configs. hardware_config import HardwareConfig
from configs.hardware_config import RTX3060Config
from configs.evaluation_config import EvaluationConfig
from data.datamodules. curriculum_datamodule import CurriculumDataModule
from evaluation.evaluators.efficiency_evaluator import EfficiencyEvaluator
from evaluation.evaluators.efficiency_evaluator import EfficiencyMetrics
from evaluation.metrics.confidence_interval import ConfidenceInterval
from evaluation.protocols.benchmark_protocol import BenchmarkProtocol
from experiments.ablation_studies.ablation_configs import AblationConfig
from experiments. ablation_studies. ablation_configs import AblationType
from experiments.ablation_studies.ablation_configs import create_ablation_configs
from models.backbones.evolvable_cnn import EvolvableCNN
from training.callbacks.evolution_callback import EvolutionCallback
from training.callbacks.nase_callback import NASECallback
from training.callbacks.curriculum_callback import CurriculumCallback
from training.lightning_modules.sseg_module import SSEGModule


@dataclass
class FewShotResult: 
    num_ways: int
    num_shots: int
    mean_accuracy:  float
    std:  float
    margin:  float
    ci_lower: float
    ci_upper:  float


@dataclass
class AblationResult: 
    config_name: str
    ablation_type: str
    description: str
    
    few_shot_results:  list[FewShotResult]
    efficiency_metrics: EfficiencyMetrics
    
    training_time_seconds: float
    final_num_blocks: int
    final_num_params: int
    final_feature_dim: int
    num_mutations: int
    
    def to_dict(self) -> dict:
        return {
            "config_name": self.config_name,
            "ablation_type":  self.ablation_type,
            "description": self.description,
            "few_shot_results":  [
                {
                    "num_ways":  fs.num_ways,
                    "num_shots": fs. num_shots,
                    "mean_accuracy": fs.mean_accuracy,
                    "std": fs.std,
                    "margin": fs.margin,
                    "ci_lower": fs.ci_lower,
                    "ci_upper":  fs.ci_upper,
                }
                for fs in self.few_shot_results
            ],
            "efficiency":  {
                "num_parameters": self.efficiency_metrics.num_parameters,
                "params_millions": self.efficiency_metrics.params_millions,
                "flops":  self.efficiency_metrics.flops,
                "flops_giga": self.efficiency_metrics.flops_giga,
                "inference_time_ms": self.efficiency_metrics.inference_time_ms,
                "memory_mb": self. efficiency_metrics.memory_mb,
            },
            "training":  {
                "training_time_seconds":  self.training_time_seconds,
                "final_num_blocks": self.final_num_blocks,
                "final_num_params": self.final_num_params,
                "final_feature_dim": self.final_feature_dim,
                "num_mutations": self.num_mutations,
            },
        }


class AblationRunner: 
    
    def __init__(
        self,
        output_dir: Path,
        data_root: Path,
        hardware_config: Optional[HardwareConfig] = None,
        seed:  int = 42,
    ):
        self._output_dir = output_dir
        self._data_root = data_root
        self._hardware_config = hardware_config or RTX3060Config()
        self._seed = seed
        
        self._output_dir.mkdir(parents=True, exist_ok=True)
        self._results:  list[AblationResult] = []
    
    def _create_base_config(self, ablation_config: AblationConfig) -> BaseConfig:
        paths = PathConfig(
            root=self._output_dir,
            data=self._data_root,
            outputs=self._output_dir / ablation_config.name,
            checkpoints=self._output_dir / ablation_config.name / "checkpoints",
            logs=self._output_dir / ablation_config. name / "logs",
            results=self._output_dir / ablation_config.name / "results",
        )
        
        from configs.evolution_config import EvolutionConfig
        
        evolution_config = EvolutionConfig(
            seed_network=ablation_config.seed_network,
            growth=ablation_config.growth,
            nase=ablation_config.nase,
        )
        
        return BaseConfig(
            experiment_name=ablation_config.name,
            seed=self._seed,
            paths=paths,
            hardware=self._hardware_config,
            curriculum=ablation_config.curriculum,
            evolution=evolution_config,
            ssl=ablation_config.ssl,
            evaluation=EvaluationConfig(),
        )
    
    def _create_callbacks(self, ablation_config: AblationConfig) -> list[pl. Callback]:
        callbacks = []
        
        if ablation_config. enable_sseg: 
            from configs.evolution_config import EvolutionConfig
            
            evolution_config = EvolutionConfig(
                seed_network=ablation_config.seed_network,
                growth=ablation_config. growth,
                nase=ablation_config.nase,
            )
            callbacks.append(EvolutionCallback(evolution_config))
        
        if ablation_config.enable_nase: 
            callbacks.append(NASECallback(ablation_config.nase))
        
        if ablation_config.enable_curriculum:
            callbacks.append(CurriculumCallback())
        
        return callbacks
    
    def run_single_ablation(
        self,
        ablation_config: AblationConfig,
        max_epochs: int = 100,
    ) -> AblationResult:
        pl.seed_everything(self._seed)
        
        base_config = self._create_base_config(ablation_config)
        
        module = SSEGModule(base_config)
        
        datamodule = CurriculumDataModule(
            curriculum_config=ablation_config.curriculum,
            hardware_config=self._hardware_config,
            seed=self._seed,
        )
        
        callbacks = self._create_callbacks(ablation_config)
        
        trainer = pl. Trainer(
            accelerator=self._hardware_config.accelerator,
            devices=self._hardware_config.devices,
            precision=self._hardware_config.precision,
            max_epochs=max_epochs,
            accumulate_grad_batches=self._hardware_config.gradient_accumulation_steps,
            gradient_clip_val=self._hardware_config.gradient_clip_val,
            callbacks=callbacks,
            enable_progress_bar=True,
            logger=False,
        )
        
        start_time = time. perf_counter()
        
        if ablation_config. ablation_type != AblationType. SEED_ONLY: 
            trainer.fit(module, datamodule)
        
        training_time = time. perf_counter() - start_time
        
        num_mutations = 0
        for callback in callbacks:
            if isinstance(callback, EvolutionCallback):
                num_mutations = len(callback.architecture_tracker.mutation_history)
        
        architecture_summary = module.backbone.get_architecture_summary()
        
        from configs.evaluation_config import FewShotConfig
        
        few_shot_config = FewShotConfig()
        benchmark = BenchmarkProtocol(
            model=module.backbone,
            few_shot_config=few_shot_config,
            device="cuda" if self._hardware_config.device == "cuda" else "cpu",
        )
        
        benchmark_result = benchmark.run_benchmark(
            method_name=ablation_config. name,
            dataset_name="MiniImageNet",
            data_root=self._data_root / "minimagenet",
        )
        
        few_shot_results = [
            FewShotResult(
                num_ways=fs.num_ways,
                num_shots=fs. num_shots,
                mean_accuracy=fs.confidence_interval.mean,
                std=fs.confidence_interval. std,
                margin=fs.confidence_interval.margin,
                ci_lower=fs.confidence_interval.lower,
                ci_upper=fs.confidence_interval. upper,
            )
            for fs in benchmark_result.few_shot_results
        ]
        
        result = AblationResult(
            config_name=ablation_config.name,
            ablation_type=ablation_config.ablation_type. name,
            description=ablation_config. description,
            few_shot_results=few_shot_results,
            efficiency_metrics=benchmark_result.efficiency_metrics,
            training_time_seconds=training_time,
            final_num_blocks=architecture_summary["num_blocks"],
            final_num_params=architecture_summary["total_params"],
            final_feature_dim=architecture_summary["feature_dim"],
            num_mutations=num_mutations,
        )
        
        self._results. append(result)
        
        return result
    
    def run_all_ablations(self, max_epochs: int = 100) -> list[AblationResult]:
        configs = create_ablation_configs()
        
        for ablation_type, ablation_config in configs.items():
            result = self.run_single_ablation(ablation_config, max_epochs)
        
        self._save_all_results()
        
        return self._results
    
    def run_selected_ablations(
        self,
        ablation_types: list[AblationType],
        max_epochs: int = 100,
    ) -> list[AblationResult]: 
        configs = create_ablation_configs()
        
        for ablation_type in ablation_types: 
            if ablation_type in configs:
                result = self.run_single_ablation(configs[ablation_type], max_epochs)
        
        self._save_all_results()
        
        return self._results
    
    def _save_all_results(self) -> None:
        results_path = self._output_dir / "ablation_results.json"
        
        all_results = [result.to_dict() for result in self._results]
        
        with open(results_path, "w") as f:
            json.dump(all_results, f, indent=2)
        
        self._generate_summary_table()
    
    def _generate_summary_table(self) -> None:
        table_path = self._output_dir / "ablation_summary. md"
        
        lines = [
            "# Ablation Study Results",
            "",
            "| Config | SSEG | NASE | Curriculum | Distill | 1-shot | 5-shot | Params | FLOPs |",
            "|--------|------|------|------------|---------|--------|--------|--------|-------|",
        ]
        
        configs = create_ablation_configs()
        
        for result in self._results:
            ablation_type = AblationType[result.ablation_type]
            config = configs[ablation_type]
            
            sseg = "Y" if config.enable_sseg else "-"
            nase = "Y" if config.enable_nase else "-"
            curriculum = "Y" if config.enable_curriculum else "-"
            distill = "Y" if config.enable_distillation else "-"
            
            one_shot = "-"
            five_shot = "-"
            
            for fs in result.few_shot_results: 
                if fs.num_shots == 1:
                    one_shot = f"{fs.mean_accuracy:.2f}±{fs.margin:.2f}"
                elif fs.num_shots == 5:
                    five_shot = f"{fs.mean_accuracy:.2f}±{fs.margin:. 2f}"
            
            params = f"{result.efficiency_metrics.params_millions:.2f}M"
            flops = f"{result.efficiency_metrics.flops_giga:.2f}G"
            
            line = f"| {result.config_name} | {sseg} | {nase} | {curriculum} | {distill} | {one_shot} | {five_shot} | {params} | {flops} |"
            lines.append(line)
        
        with open(table_path, "w") as f:
            f.write("\n". join(lines))
    
    @property
    def results(self) -> list[AblationResult]:
        return self._results. copy()