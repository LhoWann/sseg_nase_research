from dataclasses import dataclass
from typing import Literal


@dataclass(frozen=True)
class HardwareConfig:
    device: Literal["cuda", "cpu", "mps"]
    accelerator: Literal["gpu", "cpu"]
    devices: int
    precision:  Literal["32", "16-mixed", "bf16-mixed"]
    
    batch_size: int
    num_workers: int
    pin_memory: bool
    persistent_workers: bool
    
    gradient_accumulation_steps: int
    gradient_clip_val: float
    max_memory_gb: float
    
    @property
    def effective_batch_size(self) -> int:
        return self.batch_size * self.gradient_accumulation_steps
    
    def __post_init__(self) -> None:
        if self.batch_size <= 0:
            raise ValueError("batch_size must be positive")
        
        if self.gradient_accumulation_steps <= 0:
            raise ValueError("gradient_accumulation_steps must be positive")
        
        if self.gradient_clip_val <= 0:
            raise ValueError("gradient_clip_val must be positive")



@dataclass(frozen=True)

class DefaultGPUConfig(HardwareConfig):
    device: Literal["cuda", "cpu", "mps"] = "cuda"
    accelerator: Literal["gpu", "cpu"] = "gpu"
    devices: int = 1
    precision: Literal["32", "16-mixed", "bf16-mixed"] = "16-mixed"
    batch_size: int = 64
    num_workers: int = 4
    pin_memory: bool = True
    persistent_workers: bool = True
    gradient_accumulation_steps: int = 2
    gradient_clip_val: float = 1.0
    max_memory_gb: float = 8.0


def get_hardware_config(name: str) -> HardwareConfig:
    return DefaultGPUConfig()