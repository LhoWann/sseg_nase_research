from dataclasses import dataclass
from pathlib import Path
from typing import Optional
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure
@dataclass(frozen=True)
class PlotStyle:
    figure_size: tuple[float, float] = (10, 6)
    dpi: int = 300
    font_family: str = "serif"
    font_size: int = 12
    title_size: int = 14
    legend_size: int = 10
    line_width: float = 2.0
    marker_size: float = 8.0
    grid_alpha: float = 0.3
    save_format: str = "pdf"
class EvolutionPlotter: 
    def __init__(self, output_dir: Path, style: Optional[PlotStyle] = None):
        self._output_dir = output_dir
        self._style = style or PlotStyle()
        self._output_dir.mkdir(parents=True, exist_ok=True)
        self._apply_style()
    def _apply_style(self) -> None:
        plt.rcParams.update({
            "font.family": self._style.font_family,
            "font.size": self._style.font_size,
            "axes.titlesize": self._style.title_size,
            "axes.labelsize": self._style.font_size,
            "legend.fontsize": self._style.legend_size,
            "lines.linewidth": 1.0,
            "axes.grid": True,
            "grid.alpha": self._style.grid_alpha,
            "figure.dpi": self._style.dpi,
        })
    def plot_architecture_trajectory(
        self,
        epochs: list[int],
        num_params: list[float],
        num_blocks: list[int],
        mutation_epochs: list[int],
        level_boundaries: Optional[list[int]] = None,
    ) -> Path:
        fig, ax1 = plt.subplots(figsize=self._style.figure_size)
        color_params = "#2E86AB"
        ax1.set_xlabel("Epoch")
        ax1.set_ylabel("Parameters (M)", color=color_params)
        ax1.plot(
            epochs,
            num_params,
            color=color_params,
            linewidth=self._style.line_width,
            label="Parameters",
            marker="o",
            markersize=self._style.marker_size / 2,
            markevery=max(1, len(epochs) // 20),
        )
        ax1.tick_params(axis="y", labelcolor=color_params)
        ax2 = ax1.twinx()
        color_blocks = "#A23B72"
        ax2.set_ylabel("Number of Blocks", color=color_blocks)
        ax2.plot(
            epochs,
            num_blocks,
            color=color_blocks,
            linewidth=self._style.line_width,
            linestyle="--",
            label="Blocks",
        )
        ax2.tick_params(axis="y", labelcolor=color_blocks)
        for mutation_epoch in mutation_epochs:
            ax1.axvline(
                x=mutation_epoch,
                color="#E94F37",
                linestyle=":",
                alpha=0.7,
                linewidth=1.5,
            )
        if level_boundaries:
            level_colors = ["#E8F4EA", "#FFF3CD", "#FCE4EC", "#E3F2FD"]
            level_names = ["Level 1", "Level 2", "Level 3", "Level 4"]
            prev_boundary = 0
            for i, boundary in enumerate(level_boundaries):
                if i < len(level_colors):
                    ax1.axvspan(
                        prev_boundary,
                        boundary,
                        alpha=0.3,
                        color=level_colors[i],
                        label=level_names[i] if i == 0 else None,
                    )
                prev_boundary = boundary
        ax1.set_title("Network Evolution Trajectory")
        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper left")
        fig.tight_layout()
        save_path = self._output_dir / f"evolution_trajectory.{self._style.save_format.strip()}"
        fig.savefig(save_path, dpi=self._style.dpi, bbox_inches="tight")
        plt.close(fig)
        return save_path
    def plot_channel_progression(
        self,
        epochs: list[int],
        channel_history: list[list[int]],
    ) -> Path:
        fig, ax = plt.subplots(figsize=self._style.figure_size)
        max_blocks = max(len(channels) for channels in channel_history)
        colors = plt.cm.viridis(np.linspace(0, 1, max_blocks))
        for block_idx in range(max_blocks):
            block_channels = []
            block_epochs = []
            for epoch_idx, channels in enumerate(channel_history):
                if block_idx < len(channels):
                    block_channels.append(channels[block_idx])
                    block_epochs.append(epochs[epoch_idx])
            if block_channels: 
                ax.plot(
                    block_epochs,
                    block_channels,
                    color=colors[block_idx],
                    linewidth=self._style.line_width,
                    label=f"Block {block_idx + 1}",
                    marker="s",
                    markersize=self._style.marker_size / 2,
                    markevery=max(1, len(block_epochs) // 10),
                )
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Channels")
        ax.set_title("Channel Progression per Block")
        ax.legend(loc="upper left", ncol=2)
        fig.tight_layout()
        save_path = self._output_dir / f"channel_progression.{self._style.save_format.strip()}"
        fig.savefig(save_path, dpi=self._style.dpi, bbox_inches="tight")
        plt.close(fig)
        return save_path
    def plot_mutation_distribution(
        self,
        mutation_types: list[str],
        mutation_counts: list[int],
    ) -> Path:
        fig, ax = plt.subplots(figsize=(8, 6))
        colors = ["#2E86AB", "#A23B72", "#F18F01", "#C73E1D"]
        bars = ax.bar(
            mutation_types,
            mutation_counts,
            color=colors[:  len(mutation_types)],
            edgecolor="black",
            linewidth=1.2,
        )
        for bar, count in zip(bars, mutation_counts):
            height = bar.get_height()
            ax.annotate(
                f"{count}",
                xy=(bar.get_x() + bar.get_width() / 2, height),
                xytext=(0, 3),
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontsize=self._style.font_size,
            )
        ax.set_xlabel("Mutation Type")
        ax.set_ylabel("Count")
        ax.set_title("Distribution of Evolution Mutations")
        fig.tight_layout()
        save_path = self._output_dir / f"mutation_distribution.{self._style.save_format.strip()}"
        fig.savefig(save_path, dpi=self._style.dpi, bbox_inches="tight")
        plt.close(fig)
        return save_path
