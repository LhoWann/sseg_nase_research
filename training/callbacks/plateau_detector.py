from collections import deque
from dataclasses import dataclass


@dataclass(frozen=True)
class PlateauStatus:
    is_plateau: bool
    current_ssl_loss: float
    ssl_loss_delta: float
    current_distillation_loss: float
    should_evolve: bool
    should_advance_level: bool


class PlateauDetector: 
    
    def __init__(
        self,
        window_size: int,
        plateau_threshold: float,
        distillation_gap_threshold: float,
    ):
        self._window_size = window_size
        self._plateau_threshold = plateau_threshold
        self._distillation_gap_threshold = distillation_gap_threshold
        
        self._ssl_loss_history: deque[float] = deque(maxlen=window_size)
        self._distillation_loss_history: deque[float] = deque(maxlen=window_size)
    
    def update(self, ssl_loss: float, distillation_loss: float) -> None:
        self._ssl_loss_history.append(ssl_loss)
        self._distillation_loss_history.append(distillation_loss)
    
    def check_plateau(self) -> PlateauStatus:
        if len(self._ssl_loss_history) < self._window_size:
            current_ssl = (
                self._ssl_loss_history[-1] if self._ssl_loss_history else 0.0
            )
            current_distill = (
                self._distillation_loss_history[-1]
                if self._distillation_loss_history
                else 0.0
            )
            
            return PlateauStatus(
                is_plateau=False,
                current_ssl_loss=current_ssl,
                ssl_loss_delta=float("inf"),
                current_distillation_loss=current_distill,
                should_evolve=False,
                should_advance_level=False,
            )
        
        ssl_losses = list(self._ssl_loss_history)
        ssl_delta = abs(ssl_losses[-1] - ssl_losses[0])
        
        is_plateau = ssl_delta < self._plateau_threshold
        
        current_distillation = self._distillation_loss_history[-1]
        has_distillation_gap = current_distillation > self._distillation_gap_threshold
        
        should_evolve = is_plateau and has_distillation_gap
        should_advance_level = is_plateau and not has_distillation_gap
        
        return PlateauStatus(
            is_plateau=is_plateau,
            current_ssl_loss=ssl_losses[-1],
            ssl_loss_delta=ssl_delta,
            current_distillation_loss=current_distillation,
            should_evolve=should_evolve,
            should_advance_level=should_advance_level,
        )
    
    def reset(self) -> None:
        self._ssl_loss_history.clear()
        self._distillation_loss_history.clear()