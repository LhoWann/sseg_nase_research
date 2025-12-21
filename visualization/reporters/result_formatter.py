from dataclasses import dataclass
from pathlib import Path
from typing import Optional
import json


@dataclass
class FormattedResult: 
    method_name: str
    backbone: str
    params_str: str
    flops_str: str
    one_shot_str: str
    five_shot_str: str


class ResultFormatter: 

    @staticmethod
    def format_params(num_params: int) -> str:
        if num_params >= 1e6:
            return f"{num_params / 1e6:.2f}M"
        elif num_params >= 1e3:
            return f"{num_params / 1e3:.1f}K"
        else:
            return str(num_params)

    @staticmethod
    def format_flops(flops: float) -> str:
        if flops >= 1e9:
            return f"{flops / 1e9:.2f}G"
        elif flops >= 1e6:
            return f"{flops / 1e6:.1f}M"
        else:
            return f"{flops:.0f}"

    @staticmethod
    def format_accuracy(mean:  float, margin: float) -> str:
        return f"{mean:.2f}±{margin:.2f}"

    @staticmethod
    def format_percentage(value: float) -> str:
        return f"{value:.2f}%"

    @staticmethod
    def format_time_ms(time_seconds: float) -> str:
        time_ms = time_seconds * 1000
        return f"{time_ms:.2f}ms"

    @classmethod
    def format_result(
        cls,
        method_name: str,
        backbone: str,
        num_params: int,
        flops:  float,
        one_shot_mean: float,
        one_shot_margin:  float,
        five_shot_mean:  float,
        five_shot_margin:  float,
    ) -> FormattedResult:
        return FormattedResult(
            method_name=method_name,
            backbone=backbone,
            params_str=cls.format_params(num_params),
            flops_str=cls.format_flops(flops),
            one_shot_str=cls.format_accuracy(one_shot_mean, one_shot_margin),
            five_shot_str=cls.format_accuracy(five_shot_mean, five_shot_margin),
        )

    @staticmethod
    def format_improvement(baseline:  float, ours: float) -> str:
        delta = ours - baseline
        sign = "+" if delta >= 0 else ""
        return f"{sign}{delta:.2f}"

    @staticmethod
    def format_reduction(baseline: float, ours: float) -> str:
        reduction = (1 - ours / baseline) * 100
        return f"{reduction:.1f}%"


class ResultExporter:

    def __init__(self, output_dir: Path):
        self._output_dir = output_dir
        self._output_dir.mkdir(parents=True, exist_ok=True)

    def export_to_json(self, data: dict, filename: str) -> Path:
        filepath = self._output_dir / filename

        with open(filepath, "w") as f:
            json.dump(data, f, indent=2)

        return filepath

    def export_to_csv(
        self,
        headers: list[str],
        rows: list[list[str]],
        filename: str,
    ) -> Path:
        filepath = self._output_dir / filename

        with open(filepath, "w") as f:
            f.write(",".join(headers) + "\n")

            for row in rows:
                f.write(",".join(str(v) for v in row) + "\n")

        return filepath

    def export_summary(
        self,
        experiment_name: str,
        config:  dict,
        results: dict,
        filename: str,
    ) -> Path:
        summary = {
            "experiment_name": experiment_name,
            "configuration": config,
            "results": results,
        }

        return self.export_to_json(summary, filename)