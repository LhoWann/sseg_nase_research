from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np

from visualization.plotters.evolution_plotter import PlotStyle


class LossPlotter: 

    def __init__(self, output_dir: Path, style: Optional[PlotStyle] = None):
        self._output_dir = output_dir
        self._style = style or PlotStyle()
        self._output_dir.mkdir(parents=True, exist_ok=True)
        self._apply_style()

    def _apply_style(self) -> None:
        plt.rcParams.update({
            "font.family": self._style.font_family,
            "font.size": self._style.font_size,
            "axes.titlesize":  self._style.title_size,
            "axes.labelsize": self._style.font_size,
            "legend.fontsize": self._style.legend_size,
            "axes.grid": True,
            "grid.alpha":  self._style.grid_alpha,
        })

    def plot_training_losses(
        self,
        epochs: list[int],
        ssl_loss: list[float],
        distillation_loss: list[float],
        level_boundaries: Optional[list[int]] = None,
    ) -> Path:
        fig, ax = plt.subplots(figsize=self._style.figure_size)

        ax.plot(
            epochs,
            ssl_loss,
            color="#2E86AB",
            linewidth=self._style.line_width,
            label="SSL Loss (NT-Xent)",
        )

        ax.plot(
            epochs,
            distillation_loss,
            color="#A23B72",
            linewidth=self._style.line_width,
            label="Distillation Loss",
        )

        if level_boundaries:
            level_colors = ["#E8F4EA", "#FFF3CD", "#FCE4EC", "#E3F2FD"]
            level_names = ["Basic", "Texture", "Object", "Adversarial"]
            prev_boundary = 0

            for i, boundary in enumerate(level_boundaries):
                if i < len(level_colors):
                    ax.axvspan(
                        prev_boundary,
                        boundary,
                        alpha=0.2,
                        color=level_colors[i],
                    )

                    mid_point = (prev_boundary + boundary) / 2
                    y_pos = ax.get_ylim()[1] * 0.95
                    ax.text(
                        mid_point,
                        y_pos,
                        level_names[i],
                        ha="center",
                        va="top",
                        fontsize=self._style.legend_size,
                        style="italic",
                    )

                prev_boundary = boundary

        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss")
        ax.set_title("Training Loss Progression Across Curriculum Levels")
        ax.legend(loc="upper right")

        fig.tight_layout()

        save_path = self._output_dir / f"training_losses.{self._style.save_format}"
        fig.savefig(save_path, dpi=self._style.dpi, bbox_inches="tight")
        plt.close(fig)

        return save_path

    def plot_loss_with_mutations(
        self,
        epochs: list[int],
        total_loss: list[float],
        mutation_epochs: list[int],
        plateau_epochs: Optional[list[int]] = None,
    ) -> Path:
        fig, ax = plt.subplots(figsize=self._style.figure_size)

        ax.plot(
            epochs,
            total_loss,
            color="#2E86AB",
            linewidth=self._style.line_width,
            label="Total Loss",
        )

        for mutation_epoch in mutation_epochs:
            ax.axvline(
                x=mutation_epoch,
                color="#E94F37",
                linestyle="--",
                alpha=0.7,
                linewidth=1.5,
                label="Mutation" if mutation_epoch == mutation_epochs[0] else None,
            )

        if plateau_epochs:
            for plateau_epoch in plateau_epochs:
                ax.axvline(
                    x=plateau_epoch,
                    color="#F18F01",
                    linestyle=":",
                    alpha=0.5,
                    linewidth=1.0,
                )

        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss")
        ax.set_title("Loss Curve with Evolution Triggers")
        ax.legend(loc="upper right")

        fig.tight_layout()

        save_path = self._output_dir / f"loss_with_mutations.{self._style.save_format}"
        fig.savefig(save_path, dpi=self._style.dpi, bbox_inches="tight")
        plt.close(fig)

        return save_path

    def plot_loss_comparison(
        self,
        epochs: list[int],
        loss_curves: dict[str, list[float]],
    ) -> Path:
        fig, ax = plt.subplots(figsize=self._style.figure_size)

        colors = ["#2E86AB", "#A23B72", "#F18F01", "#C73E1D", "#6A4C93"]

        for idx, (label, losses) in enumerate(loss_curves.items()):
            color = colors[idx % len(colors)]
            ax.plot(
                epochs[:  len(losses)],
                losses,
                color=color,
                linewidth=self._style.line_width,
                label=label,
            )

        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss")
        ax.set_title("Loss Comparison Across Configurations")
        ax.legend(loc="upper right")

        fig.tight_layout()

        save_path = self._output_dir / f"loss_comparison.{self._style.save_format}"
        fig.savefig(save_path, dpi=self._style.dpi, bbox_inches="tight")
        plt.close(fig)

        return save_path