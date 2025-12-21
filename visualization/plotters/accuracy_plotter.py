from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np

from visualization.plotters.evolution_plotter import PlotStyle


@dataclass
class AccuracyData:
    method_name: str
    one_shot_mean: float
    one_shot_margin: float
    five_shot_mean:  float
    five_shot_margin: float
    params_millions: float
    flops_giga: float


class AccuracyPlotter:

    def __init__(self, output_dir:  Path, style: Optional[PlotStyle] = None):
        self._output_dir = output_dir
        self._style = style or PlotStyle()
        self._output_dir.mkdir(parents=True, exist_ok=True)
        self._apply_style()

    def _apply_style(self) -> None:
        plt.rcParams.update({
            "font.family": self._style.font_family,
            "font.size": self._style.font_size,
            "axes.titlesize": self._style.title_size,
            "axes.labelsize":  self._style.font_size,
            "legend.fontsize": self._style.legend_size,
            "axes.grid": True,
            "grid.alpha":  self._style.grid_alpha,
        })

    def plot_accuracy_vs_complexity(
        self,
        data: list[AccuracyData],
        highlight_method: str = "SSEG-NASE",
    ) -> Path:
        fig, ax = plt.subplots(figsize=self._style.figure_size)

        for entry in data:
            is_ours = entry.method_name == highlight_method
            color = "#E94F37" if is_ours else "#2E86AB"
            marker = "★" if is_ours else "o"
            size = 200 if is_ours else 100

            ax.scatter(
                entry.flops_giga,
                entry.five_shot_mean,
                s=size,
                c=color,
                marker=marker if not is_ours else "o",
                edgecolors="black",
                linewidth=1.5,
                alpha=0.8,
                zorder=10 if is_ours else 5,
            )

            offset_x = 0.05
            offset_y = 0.5
            ax.annotate(
                entry.method_name,
                (entry.flops_giga, entry.five_shot_mean),
                textcoords="offset points",
                xytext=(5, 5),
                fontsize=self._style.legend_size,
                fontweight="bold" if is_ours else "normal",
            )

        ax.set_xlabel("FLOPs (G)")
        ax.set_ylabel("5-way 5-shot Accuracy (%)")
        ax.set_title("Accuracy vs Computational Complexity")

        fig.tight_layout()

        save_path = self._output_dir / f"accuracy_vs_complexity. {self._style.save_format}"
        fig.savefig(save_path, dpi=self._style.dpi, bbox_inches="tight")
        plt.close(fig)

        return save_path

    def plot_ablation_bar_chart(
        self,
        config_names: list[str],
        one_shot_acc: list[float],
        one_shot_margin:  list[float],
        five_shot_acc: list[float],
        five_shot_margin: list[float],
    ) -> Path:
        fig, ax = plt.subplots(figsize=(12, 6))

        x = np.arange(len(config_names))
        width = 0.35

        bars1 = ax.bar(
            x - width / 2,
            one_shot_acc,
            width,
            yerr=one_shot_margin,
            label="5-way 1-shot",
            color="#2E86AB",
            capsize=4,
            edgecolor="black",
            linewidth=1,
        )

        bars2 = ax.bar(
            x + width / 2,
            five_shot_acc,
            width,
            yerr=five_shot_margin,
            label="5-way 5-shot",
            color="#A23B72",
            capsize=4,
            edgecolor="black",
            linewidth=1,
        )

        ax.set_xlabel("Configuration")
        ax.set_ylabel("Accuracy (%)")
        ax.set_title("Ablation Study Results")
        ax.set_xticks(x)
        ax.set_xticklabels(config_names, rotation=45, ha="right")
        ax.legend(loc="upper left")

        ax.set_ylim(bottom=0)

        fig.tight_layout()

        save_path = self._output_dir / f"ablation_bar_chart.{self._style.save_format}"
        fig.savefig(save_path, dpi=self._style.dpi, bbox_inches="tight")
        plt.close(fig)

        return save_path

    def plot_shot_comparison(
        self,
        methods: list[str],
        one_shot_acc: list[float],
        five_shot_acc:  list[float],
    ) -> Path:
        fig, ax = plt.subplots(figsize=self._style.figure_size)

        x = np.arange(len(methods))

        ax.plot(
            x,
            one_shot_acc,
            marker="o",
            linewidth=self._style.line_width,
            markersize=self._style.marker_size,
            color="#2E86AB",
            label="1-shot",
        )

        ax.plot(
            x,
            five_shot_acc,
            marker="s",
            linewidth=self._style.line_width,
            markersize=self._style.marker_size,
            color="#A23B72",
            label="5-shot",
        )

        ax.set_xticks(x)
        ax.set_xticklabels(methods, rotation=45, ha="right")
        ax.set_xlabel("Method")
        ax.set_ylabel("Accuracy (%)")
        ax.set_title("1-shot vs 5-shot Performance Comparison")
        ax.legend(loc="lower right")

        fig.tight_layout()

        save_path = self._output_dir / f"shot_comparison.{self._style.save_format}"
        fig.savefig(save_path, dpi=self._style.dpi, bbox_inches="tight")
        plt.close(fig)

        return save_path