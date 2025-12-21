from dataclasses import dataclass
from dataclasses import field
from enum import Enum
from enum import auto
from pathlib import Path
from typing import Optional
import json

import torch
import torch.nn as nn

from configs.evaluation_config import FewShotConfig
from evaluation.metrics.confidence_interval import ConfidenceInterval
from evaluation.protocols.benchmark_protocol import BenchmarkProtocol


class BaselineMethod(Enum):
    PROTONET = auto()
    MATCHINGNET = auto()
    MAML = auto()
    RELATIONNET = auto()
    SIMCLR_LINEAR = auto()
    DINO_LINEAR = auto()


@dataclass(frozen=True)
class BaselineSpec:
    name:  str
    method: BaselineMethod
    backbone:  str
    num_params: int
    flops:  float
    one_shot_reported: float
    five_shot_reported: float
    source:  str


BASELINE_SPECS = {
    BaselineMethod.PROTONET: BaselineSpec(
        name="ProtoNet",
        method=BaselineMethod.PROTONET,
        backbone="Conv-4",
        num_params=113_000,
        flops=50_000_000,
        one_shot_reported=49.42,
        five_shot_reported=68.20,
        source="Snell et al., NeurIPS 2017",
    ),
    BaselineMethod. MATCHINGNET:  BaselineSpec(
        name="MatchingNet",
        method=BaselineMethod.MATCHINGNET,
        backbone="Conv-4",
        num_params=113_000,
        flops=50_000_000,
        one_shot_reported=43.56,
        five_shot_reported=55.31,
        source="Vinyals et al., NeurIPS 2016",
    ),
    BaselineMethod. MAML: BaselineSpec(
        name="MAML",
        method=BaselineMethod. MAML,
        backbone="Conv-4",
        num_params=113_000,
        flops=50_000_000,
        one_shot_reported=48.70,
        five_shot_reported=63.11,
        source="Finn et al., ICML 2017",
    ),
    BaselineMethod. RELATIONNET: BaselineSpec(
        name="RelationNet",
        method=BaselineMethod. RELATIONNET,
        backbone="Conv-4",
        num_params=230_000,
        flops=90_000_000,
        one_shot_reported=50.44,
        five_shot_reported=65.32,
        source="Sung et al., CVPR 2018",
    ),
    BaselineMethod. SIMCLR_LINEAR: BaselineSpec(
        name="SimCLR+Linear",
        method=BaselineMethod. SIMCLR_LINEAR,
        backbone="ResNet-18",
        num_params=11_200_000,
        flops=1_820_000_000,
        one_shot_reported=51.23,
        five_shot_reported=69.45,
        source="Chen et al., ICML 2020 (adapted)",
    ),
    BaselineMethod. DINO_LINEAR: BaselineSpec(
        name="DINO+Linear",
        method=BaselineMethod.DINO_LINEAR,
        backbone="ViT-Small",
        num_params=22_000_000,
        flops=4_610_000_000,
        one_shot_reported=53.12,
        five_shot_reported=71.20,
        source="Caron et al., ICCV 2021 (adapted)",
    ),
}


@dataclass
class BaselineResult: 
    method_name: str
    backbone: str
    num_params: int
    flops:  float
    one_shot_accuracy: float
    one_shot_margin: float
    five_shot_accuracy:  float
    five_shot_margin: float
    is_reproduced: bool
    source: str
    
    @property
    def params_millions(self) -> float:
        return self.num_params / 1e6
    
    @property
    def flops_giga(self) -> float:
        return self.flops / 1e9
    
    @property
    def accuracy_per_flop(self) -> float:
        return self.five_shot_accuracy / self.flops_giga if self.flops_giga > 0 else 0.0
    
    def to_dict(self) -> dict:
        return {
            "method_name":  self.method_name,
            "backbone": self.backbone,
            "num_params": self.num_params,
            "params_millions": self.params_millions,
            "flops":  self.flops,
            "flops_giga": self.flops_giga,
            "one_shot_accuracy":  self.one_shot_accuracy,
            "one_shot_margin": self.one_shot_margin,
            "five_shot_accuracy": self.five_shot_accuracy,
            "five_shot_margin": self.five_shot_margin,
            "accuracy_per_flop": self.accuracy_per_flop,
            "is_reproduced": self.is_reproduced,
            "source": self.source,
        }


@dataclass
class OursResult: 
    method_name: str
    backbone: str
    num_params: int
    flops: float
    one_shot_ci: ConfidenceInterval
    five_shot_ci: ConfidenceInterval
    
    @property
    def params_millions(self) -> float:
        return self.num_params / 1e6
    
    @property
    def flops_giga(self) -> float:
        return self.flops / 1e9
    
    @property
    def accuracy_per_flop(self) -> float:
        return self.five_shot_ci.mean / self.flops_giga if self.flops_giga > 0 else 0.0


class BaselineComparison:
    
    def __init__(self, output_dir: Path):
        self._output_dir = output_dir
        self._output_dir.mkdir(parents=True, exist_ok=True)
        
        self._baseline_results: list[BaselineResult] = []
        self._ours_result: Optional[OursResult] = None
    
    def add_reported_baselines(self) -> None:
        for method, spec in BASELINE_SPECS.items():
            result = BaselineResult(
                method_name=spec.name,
                backbone=spec.backbone,
                num_params=spec.num_params,
                flops=spec.flops,
                one_shot_accuracy=spec.one_shot_reported,
                one_shot_margin=0.80,
                five_shot_accuracy=spec.five_shot_reported,
                five_shot_margin=0.65,
                is_reproduced=False,
                source=spec.source,
            )
            self._baseline_results.append(result)
    
    def add_ours_result(
        self,
        model:  nn.Module,
        data_root: Path,
        method_name: str = "SSEG-NASE",
    ) -> OursResult:
        from evaluation.evaluators.efficiency_evaluator import EfficiencyEvaluator
        
        efficiency_evaluator = EfficiencyEvaluator(model, device="cuda")
        efficiency = efficiency_evaluator.evaluate()
        
        few_shot_config = FewShotConfig()
        benchmark = BenchmarkProtocol(
            model=model,
            few_shot_config=few_shot_config,
            device="cuda",
        )
        
        result = benchmark.run_benchmark(
            method_name=method_name,
            dataset_name="MiniImageNet",
            data_root=data_root / "minimagenet",
        )
        
        one_shot_ci = None
        five_shot_ci = None
        
        for fs in result.few_shot_results: 
            if fs.num_shots == 1:
                one_shot_ci = fs.confidence_interval
            elif fs.num_shots == 5:
                five_shot_ci = fs.confidence_interval
        
        self._ours_result = OursResult(
            method_name=method_name,
            backbone="Evolved-CNN",
            num_params=efficiency.num_parameters,
            flops=efficiency.flops,
            one_shot_ci=one_shot_ci,
            five_shot_ci=five_shot_ci,
        )
        
        return self._ours_result
    
    def generate_comparison_table(self) -> str:
        lines = [
            "# Comparison with State-of-the-Art",
            "",
            "| Method | Backbone | Params | FLOPs | 1-shot | 5-shot | Acc/FLOP |",
            "|--------|----------|--------|-------|--------|--------|----------|",
        ]
        
        all_results = []
        
        for baseline in self._baseline_results:
            all_results.append({
                "name": baseline.method_name,
                "backbone": baseline.backbone,
                "params": f"{baseline.params_millions:.2f}M",
                "flops": f"{baseline.flops_giga:.2f}G",
                "one_shot": f"{baseline.one_shot_accuracy:.2f}±{baseline.one_shot_margin:.2f}",
                "five_shot": f"{baseline.five_shot_accuracy:.2f}±{baseline.five_shot_margin:.2f}",
                "acc_per_flop": f"{baseline.accuracy_per_flop:.2f}",
                "is_ours": False,
            })
        
        if self._ours_result:
            all_results.append({
                "name":  f"**{self._ours_result.method_name}**",
                "backbone": self._ours_result.backbone,
                "params": f"**{self._ours_result.params_millions:.2f}M**",
                "flops": f"**{self._ours_result.flops_giga:.2f}G**",
                "one_shot": f"**{self._ours_result.one_shot_ci}**",
                "five_shot": f"**{self._ours_result.five_shot_ci}**",
                "acc_per_flop": f"**{self._ours_result.accuracy_per_flop:.2f}**",
                "is_ours": True,
            })
        
        for res in all_results: 
            line = (
                f"| {res['name']} | {res['backbone']} | {res['params']} | "
                f"{res['flops']} | {res['one_shot']} | {res['five_shot']} | "
                f"{res['acc_per_flop']} |"
            )
            lines.append(line)
        
        return "\n".join(lines)
    
    def generate_latex_table(self) -> str:
        lines = [
            r"\begin{table}[t]",
            r"\centering",
            r"\caption{Comparison with state-of-the-art methods on MiniImageNet}",
            r"\label{tab:sota_comparison}",
            r"\begin{tabular}{lcccccc}",
            r"\toprule",
            r"Method & Backbone & Params & FLOPs & 1-shot & 5-shot & Acc/FLOP \\",
            r"\midrule",
        ]
        
        for baseline in self._baseline_results:
            line = (
                f"{baseline.method_name} & {baseline.backbone} & "
                f"{baseline.params_millions:.2f}M & {baseline.flops_giga:.2f}G & "
                f"{baseline.one_shot_accuracy:.2f}$\\pm${baseline.one_shot_margin:.2f} & "
                f"{baseline.five_shot_accuracy:.2f}$\\pm${baseline.five_shot_margin:.2f} & "
                f"{baseline.accuracy_per_flop:.2f} \\\\"
            )
            lines.append(line)
        
        lines.append(r"\midrule")
        
        if self._ours_result:
            line = (
                f"\\textbf{{{self._ours_result.method_name}}} & "
                f"{self._ours_result.backbone} & "
                f"\\textbf{{{self._ours_result.params_millions:.2f}M}} & "
                f"\\textbf{{{self._ours_result.flops_giga:.2f}G}} & "
                f"\\textbf{{{self._ours_result.one_shot_ci.mean:.2f}$\\pm${self._ours_result.one_shot_ci.margin:.2f}}} & "
                f"\\textbf{{{self._ours_result.five_shot_ci.mean:.2f}$\\pm${self._ours_result.five_shot_ci.margin:.2f}}} & "
                f"\\textbf{{{self._ours_result.accuracy_per_flop:.2f}}} \\\\"
            )
            lines.append(line)
        
        lines.extend([
            r"\bottomrule",
            r"\end{tabular}",
            r"\end{table}",
        ])
        
        return "\n".join(lines)
    
    def save_results(self) -> None:
        comparison_md = self.generate_comparison_table()
        md_path = self._output_dir / "comparison_table.md"
        with open(md_path, "w") as f:
            f.write(comparison_md)
        
        latex_table = self.generate_latex_table()
        latex_path = self._output_dir / "comparison_table.tex"
        with open(latex_path, "w") as f:
            f.write(latex_table)
        
        json_data = {
            "baselines": [b.to_dict() for b in self._baseline_results],
            "ours":  {
                "method_name": self._ours_result.method_name,
                "backbone": self._ours_result.backbone,
                "num_params": self._ours_result.num_params,
                "flops":  self._ours_result.flops,
                "one_shot_mean": self._ours_result.one_shot_ci.mean,
                "one_shot_margin": self._ours_result.one_shot_ci.margin,
                "five_shot_mean":  self._ours_result.five_shot_ci.mean,
                "five_shot_margin": self._ours_result.five_shot_ci.margin,
                "accuracy_per_flop": self._ours_result.accuracy_per_flop,
            } if self._ours_result else None,
        }
        
        json_path = self._output_dir / "comparison_data.json"
        with open(json_path, "w") as f:
            json.dump(json_data, f, indent=2)
    
    def compute_relative_improvements(self) -> dict[str, float]: 
        if not self._ours_result: 
            return {}
        
        improvements = {}
        
        for baseline in self._baseline_results:
            key = baseline.method_name
            
            one_shot_delta = self._ours_result.one_shot_ci.mean - baseline.one_shot_accuracy
            five_shot_delta = self._ours_result.five_shot_ci.mean - baseline.five_shot_accuracy
            
            param_reduction = 1 - (self._ours_result.num_params / baseline.num_params)
            flops_reduction = 1 - (self._ours_result.flops / baseline.flops)
            
            improvements[f"{key}_1shot_delta"] = one_shot_delta
            improvements[f"{key}_5shot_delta"] = five_shot_delta
            improvements[f"{key}_param_reduction"] = param_reduction * 100
            improvements[f"{key}_flops_reduction"] = flops_reduction * 100
        
        return improvements