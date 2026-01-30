from dataclasses import dataclass
import time
from typing import Optional

import torch
import torch.nn as nn
from torch import Tensor


@dataclass
class EfficiencyMetrics:
    num_parameters: int
    params_millions: float
    flops: int
    flops_giga: float
    inference_time_ms: float
    memory_mb: float
    
    def to_dict(self) -> dict:
        return {
            "num_parameters": self.num_parameters,
            "params_millions": self.params_millions,
            "flops": self.flops,
            "flops_giga": self.flops_giga,
            "inference_time_ms": self.inference_time_ms,
            "memory_mb": self.memory_mb,
        }

class EfficiencyEvaluator:
    
    def __init__(
        self,
        model: nn.Module,
        device: str = "cuda",
        input_size: tuple[int, int, int, int] = (1, 3, 84, 84),
    ):
        self._device = device
        self._model = model.to(self._device)
        self._input_size = input_size
        self._model.eval()
    
    def count_parameters(self) -> tuple[int, float]:
        num_params = sum(p.numel() for p in self._model.parameters())
        params_millions = num_params / 1e6
        return num_params, params_millions
    
    def estimate_flops(self) -> tuple[int, float]:
        self._model = self._model.to(self._device)
        flops = 0

        def hook_fn(module, input, output):
            nonlocal flops
            if isinstance(module, nn.Conv2d):
                batch_size = input[0].size(0)
                output_h, output_w = output.size(2), output.size(3)
                kernel_ops = module.kernel_size[0] * module.kernel_size[1]
                input_channels = module.in_channels // module.groups
                flops += batch_size * output_h * output_w * input_channels * kernel_ops * module.out_channels
            elif isinstance(module, nn.Linear):
                batch_size = input[0].size(0)
                flops += batch_size * module.in_features * module.out_features

        hooks = []
        for module in self._model.modules():
            if isinstance(module, (nn.Conv2d, nn.Linear)):
                hooks.append(module.register_forward_hook(hook_fn))

        dummy_input = torch.randn(*self._input_size)
        dummy_input = dummy_input.to(next(self._model.parameters()).device)

        with torch.no_grad():
            self._model(dummy_input)

        for hook in hooks:
            hook.remove()

        return flops, flops / 1e9
    
    def measure_inference_time(
        self,
        num_warmup: int = 10,
        num_runs: int = 100,
    ) -> float:
        dummy_input = torch.randn(*self._input_size, device=self._device)
        
        # Warmup
        with torch.no_grad():
            for _ in range(num_warmup):
                self._model(dummy_input)
        
        if self._device == "cuda":
            torch.cuda.synchronize()
        
        # Measure
        start_time = time.perf_counter()
        
        with torch.no_grad():
            for _ in range(num_runs):
                self._model(dummy_input)
        
        if self._device == "cuda":
            torch.cuda.synchronize()
        
        end_time = time.perf_counter()
        
        avg_time_ms = (end_time - start_time) / num_runs * 1000
        return avg_time_ms
    
    def measure_memory(self) -> float:
        param_memory = sum(p.numel() * p.element_size() for p in self._model.parameters())
        buffer_memory = sum(b.numel() * b.element_size() for b in self._model.buffers())
        total_memory_mb = (param_memory + buffer_memory) / (1024 * 1024)
        return total_memory_mb
    
    def evaluate(self) -> EfficiencyMetrics:
        num_params, params_millions = self.count_parameters()
        flops, flops_giga = self.estimate_flops()
        inference_time_ms = self.measure_inference_time()
        memory_mb = self.measure_memory()
        
        return EfficiencyMetrics(
            num_parameters=num_params,
            params_millions=params_millions,
            flops=flops,
            flops_giga=flops_giga,
            inference_time_ms=inference_time_ms,
            memory_mb=memory_mb,
        )
