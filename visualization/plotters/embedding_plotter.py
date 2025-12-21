from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import NDArray

from visualization.plotters.evolution_plotter import PlotStyle


class EmbeddingPlotter:

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
        })

    def plot_tsne(
        self,
        embeddings: NDArray,
        labels: NDArray,
        class_names: Optional[list[str]] = None,
        title: str = "t-SNE Visualization of Feature Embeddings",
        perplexity: int = 30,
    ) -> Path:
        from sklearn.manifold import TSNE

        tsne = TSNE(
            n_components=2,
            random_state=42,
            perplexity=perplexity,
            n_iter=1000,
        )
        embeddings_2d = tsne.fit_transform(embeddings)

        fig, ax = plt.subplots(figsize=self._style.figure_size)

        unique_labels = np.unique(labels)
        num_classes = len(unique_labels)
        colors = plt.cm.tab10(np.linspace(0, 1, min(num_classes, 10)))

        for idx, label in enumerate(unique_labels):
            mask = labels == label
            label_name = class_names[idx] if class_names else f"Class {label}"

            ax.scatter(
                embeddings_2d[mask, 0],
                embeddings_2d[mask, 1],
                c=[colors[idx % len(colors)]],
                label=label_name,
                alpha=0.7,
                s=30,
                edgecolors="white",
                linewidth=0.5,
            )

        ax.set_xlabel("t-SNE Dimension 1")
        ax.set_ylabel("t-SNE Dimension 2")
        ax.set_title(title)

        if num_classes <= 10:
            ax.legend(loc="best", markerscale=1.5)

        fig.tight_layout()

        save_path = self._output_dir / f"tsne_embeddings.{self._style.save_format}"
        fig.savefig(save_path, dpi=self._style.dpi, bbox_inches="tight")
        plt.close(fig)

        return save_path

    def plot_umap(
        self,
        embeddings:  NDArray,
        labels: NDArray,
        class_names: Optional[list[str]] = None,
        title: str = "UMAP Visualization of Feature Embeddings",
        n_neighbors: int = 15,
        min_dist: float = 0.1,
    ) -> Path:
        try:
            import umap
        except ImportError:
            raise ImportError("umap-learn package required for UMAP visualization")

        reducer = umap.UMAP(
            n_components=2,
            random_state=42,
            n_neighbors=n_neighbors,
            min_dist=min_dist,
        )
        embeddings_2d = reducer.fit_transform(embeddings)

        fig, ax = plt.subplots(figsize=self._style.figure_size)

        unique_labels = np.unique(labels)
        num_classes = len(unique_labels)
        colors = plt.cm.tab10(np.linspace(0, 1, min(num_classes, 10)))

        for idx, label in enumerate(unique_labels):
            mask = labels == label
            label_name = class_names[idx] if class_names else f"Class {label}"

            ax.scatter(
                embeddings_2d[mask, 0],
                embeddings_2d[mask, 1],
                c=[colors[idx % len(colors)]],
                label=label_name,
                alpha=0.7,
                s=30,
                edgecolors="white",
                linewidth=0.5,
            )

        ax.set_xlabel("UMAP Dimension 1")
        ax.set_ylabel("UMAP Dimension 2")
        ax.set_title(title)

        if num_classes <= 10:
            ax.legend(loc="best", markerscale=1.5)

        fig.tight_layout()

        save_path = self._output_dir / f"umap_embeddings.{self._style.save_format}"
        fig.savefig(save_path, dpi=self._style.dpi, bbox_inches="tight")
        plt.close(fig)

        return save_path

    def plot_embedding_comparison(
        self,
        embeddings_before: NDArray,
        embeddings_after: NDArray,
        labels: NDArray,
    ) -> Path:
        from sklearn.manifold import TSNE

        fig, axes = plt.subplots(1, 2, figsize=(14, 6))

        tsne = TSNE(n_components=2, random_state=42, perplexity=30)

        emb_before_2d = tsne.fit_transform(embeddings_before)
        emb_after_2d = tsne.fit_transform(embeddings_after)

        unique_labels = np.unique(labels)
        colors = plt.cm.tab10(np.linspace(0, 1, min(len(unique_labels), 10)))

        for idx, label in enumerate(unique_labels):
            mask = labels == label

            axes[0].scatter(
                emb_before_2d[mask, 0],
                emb_before_2d[mask, 1],
                c=[colors[idx % len(colors)]],
                alpha=0.7,
                s=30,
            )

            axes[1].scatter(
                emb_after_2d[mask, 0],
                emb_after_2d[mask, 1],
                c=[colors[idx % len(colors)]],
                alpha=0.7,
                s=30,
            )

        axes[0].set_title("Before Evolution")
        axes[0].set_xlabel("t-SNE Dimension 1")
        axes[0].set_ylabel("t-SNE Dimension 2")

        axes[1].set_title("After Evolution")
        axes[1].set_xlabel("t-SNE Dimension 1")
        axes[1].set_ylabel("t-SNE Dimension 2")

        fig.suptitle("Feature Embedding Quality Comparison", fontsize=14)
        fig.tight_layout()

        save_path = self._output_dir / f"embedding_comparison. {self._style.save_format}"
        fig.savefig(save_path, dpi=self._style.dpi, bbox_inches="tight")
        plt.close(fig)

        return save_path