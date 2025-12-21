from dataclasses import dataclass

import torch
import torch.nn as nn


@dataclass
class ComplexityMetrics:
    num_parameters: int
    trainable_parameters: int
    flops: int
    memory_bytes: int
    
    @property
    def params_millions(self) -> float:
        return self.num_parameters / 1e6
    
    @property
    def flops_giga(self) -> float:
        return self.flops / 1e9
    
    @property
    def memory_mb(self) -> float:
        return self.memory_bytes / (1024 * 1024)


def count_parameters(model: nn.Module) -> tuple[int, int]:
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


def estimate_flops(
    model: nn.Module,
    input_size: tuple[int, int, int, int],
    device: str = "cpu",
) -> int:
    """Estimate FLOPs for a model with given input size."""
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
    for module in model.modules():
        if isinstance(module, (nn.Conv2d, nn.Linear)):
            hooks.append(module.register_forward_hook(hook_fn))
    
    model = model.to(device)
    dummy_input = torch.randn(*input_size, device=device)
    
    with torch.no_grad():
        model(dummy_input)
    
    for hook in hooks:
        hook.remove()
    
    return flops


def compute_complexity_metrics(
    model: nn.Module,
    input_size: tuple[int, int, int, int] = (1, 3, 84, 84),
    device: str = "cpu",
) -> ComplexityMetrics:
    total_params, trainable_params = count_parameters(model)
    flops = estimate_flops(model, input_size, device)
    
    memory = sum(p.numel() * p.element_size() for p in model.parameters())
    
    return ComplexityMetrics(
        num_parameters=total_params,
        trainable_parameters=trainable_params,
        flops=flops,
        memory_bytes=memory,
    )
