from pathlib import Path
from typing import Optional

import torch
from torch import Tensor
from torch.utils.tensorboard import SummaryWriter


class TensorBoardLogger:

    def __init__(self, log_dir: Path, experiment_name: str):
        self._log_dir = log_dir / experiment_name
        self._log_dir.mkdir(parents=True, exist_ok=True)
        self._writer = SummaryWriter(log_dir=str(self._log_dir))
        self._step = 0

    def log_scalar(self, tag: str, value: float, step: Optional[int] = None) -> None:
        current_step = step if step is not None else self._step
        self._writer.add_scalar(tag, value, current_step)

    def log_scalars(
        self,
        main_tag: str,
        tag_scalar_dict: dict[str, float],
        step: Optional[int] = None,
    ) -> None:
        current_step = step if step is not None else self._step
        self._writer.add_scalars(main_tag, tag_scalar_dict, current_step)

    def log_histogram(
        self,
        tag: str,
        values: Tensor,
        step:  Optional[int] = None,
    ) -> None:
        current_step = step if step is not None else self._step
        self._writer.add_histogram(tag, values, current_step)

    def log_image(
        self,
        tag: str,
        image:  Tensor,
        step: Optional[int] = None,
    ) -> None:
        current_step = step if step is not None else self._step
        self._writer.add_image(tag, image, current_step)

    def log_architecture(
        self,
        num_blocks: int,
        num_params: int,
        feature_dim: int,
        step: Optional[int] = None,
    ) -> None:
        current_step = step if step is not None else self._step

        self._writer.add_scalar("architecture/num_blocks", num_blocks, current_step)
        self._writer.add_scalar("architecture/num_params", num_params, current_step)
        self._writer.add_scalar("architecture/feature_dim", feature_dim, current_step)

    def log_training_metrics(
        self,
        ssl_loss: float,
        distillation_loss: float,
        total_loss: float,
        learning_rate: float,
        step: Optional[int] = None,
    ) -> None:
        current_step = step if step is not None else self._step

        self._writer.add_scalar("train/ssl_loss", ssl_loss, current_step)
        self._writer.add_scalar("train/distillation_loss", distillation_loss, current_step)
        self._writer.add_scalar("train/total_loss", total_loss, current_step)
        self._writer.add_scalar("train/learning_rate", learning_rate, current_step)

    def log_evolution_event(
        self,
        mutation_type: str,
        target_layer: Optional[int],
        params_before: int,
        params_after: int,
        step: Optional[int] = None,
    ) -> None:
        current_step = step if step is not None else self._step

        text = (
            f"Mutation:  {mutation_type}\n"
            f"Target Layer: {target_layer}\n"
            f"Params Before: {params_before: ,}\n"
            f"Params After: {params_after:,}\n"
            f"Delta: {params_after - params_before: ,}"
        )

        self._writer.add_text("evolution/mutations", text, current_step)

    def log_evaluation_metrics(
        self,
        num_shots: int,
        mean_accuracy: float,
        margin:  float,
        step: Optional[int] = None,
    ) -> None:
        current_step = step if step is not None else self._step

        self._writer.add_scalar(f"eval/{num_shots}shot_accuracy", mean_accuracy, current_step)
        self._writer.add_scalar(f"eval/{num_shots}shot_margin", margin, current_step)

    def increment_step(self) -> None:
        self._step += 1

    def flush(self) -> None:
        self._writer.flush()

    def close(self) -> None:
        self._writer.close()