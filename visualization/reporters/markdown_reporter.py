from dataclasses import dataclass
from pathlib import Path
from typing import Optional
import json


@dataclass
class ExperimentSummary:
    experiment_name: str
    description: str
    final_architecture: dict
    training_metrics: dict
    evaluation_metrics: dict


class MarkdownReporter:

    def __init__(self, output_dir: Path):
        self._output_dir = output_dir
        self._output_dir. mkdir(parents=True, exist_ok=True)

    def generate_experiment_report(self, summary: ExperimentSummary) -> str:
        lines = [
            f"# Experiment Report: {summary.experiment_name}",
            "",
            "## Description",
            "",
            summary.description,
            "",
            "## Final Architecture",
            "",
            "| Property | Value |",
            "|----------|-------|",
        ]

        for key, value in summary. final_architecture.items():
            lines. append(f"| {key} | {value} |")

        lines.extend([
            "",
            "## Training Metrics",
            "",
            "| Metric | Value |",
            "|--------|-------|",
        ])

        for key, value in summary. training_metrics.items():
            if isinstance(value, float):
                lines. append(f"| {key} | {value:.4f} |")
            else:
                lines. append(f"| {key} | {value} |")

        lines.extend([
            "",
            "## Evaluation Metrics",
            "",
            "| Metric | Value |",
            "|--------|-------|",
        ])

        for key, value in summary.evaluation_metrics. items():
            if isinstance(value, float):
                lines. append(f"| {key} | {value:.2f} |")
            else:
                lines.append(f"| {key} | {value} |")

        return "\n".join(lines)

    def generate_comparison_table(
        self,
        methods: list[str],
        backbones: list[str],
        params: list[str],
        flops:  list[str],
        one_shot:  list[str],
        five_shot:  list[str],
        highlight_idx: Optional[int] = None,
    ) -> str:
        lines = [
            "# Comparison with State-of-the-Art",
            "",
            "| Method | Backbone | Params | FLOPs | 1-shot | 5-shot |",
            "|--------|----------|--------|-------|--------|--------|",
        ]

        for idx in range(len(methods)):
            method = methods[idx]
            if highlight_idx is not None and idx == highlight_idx:
                method = f"**{method}**"
                row = (
                    f"| {method} | {backbones[idx]} | **{params[idx]}** | "
                    f"**{flops[idx]}** | **{one_shot[idx]}** | **{five_shot[idx]}** |"
                )
            else:
                row = (
                    f"| {method} | {backbones[idx]} | {params[idx]} | "
                    f"{flops[idx]} | {one_shot[idx]} | {five_shot[idx]} |"
                )
            lines.append(row)

        return "\n".join(lines)

    def generate_ablation_table(
        self,
        configs: list[str],
        sseg:  list[bool],
        nase: list[bool],
        curriculum: list[bool],
        distillation: list[bool],
        one_shot:  list[str],
        five_shot:  list[str],
    ) -> str:
        lines = [
            "# Ablation Study Results",
            "",
            "| Config | SSEG | NASE | Curriculum | Distill | 1-shot | 5-shot |",
            "|--------|------|------|------------|---------|--------|--------|",
        ]

        for idx in range(len(configs)):
            sseg_str = "Y" if sseg[idx] else "-"
            nase_str = "Y" if nase[idx] else "-"
            curr_str = "Y" if curriculum[idx] else "-"
            dist_str = "Y" if distillation[idx] else "-"

            row = (
                f"| {configs[idx]} | {sseg_str} | {nase_str} | "
                f"{curr_str} | {dist_str} | {one_shot[idx]} | {five_shot[idx]} |"
            )
            lines.append(row)

        return "\n".join(lines)

    def save_report(self, content: str, filename:  str) -> Path:
        filepath = self._output_dir / filename

        with open(filepath, "w") as f:
            f.write(content)

        return filepath