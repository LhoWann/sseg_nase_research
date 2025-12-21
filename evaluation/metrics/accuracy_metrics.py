import torch
from torch import Tensor


def compute_accuracy(predictions: Tensor, targets: Tensor) -> float:
    correct = (predictions == targets).sum().item()
    total = targets.numel()
    return correct / total * 100.0


def compute_per_class_accuracy(
    predictions: Tensor,
    targets: Tensor,
    num_classes: int,
) -> dict[int, float]:
    per_class_acc = {}
    
    for class_idx in range(num_classes):
        mask = targets == class_idx
        if mask.sum() == 0:
            per_class_acc[class_idx] = 0.0
        else:
            correct = (predictions[mask] == targets[mask]).sum().item()
            total = mask.sum().item()
            per_class_acc[class_idx] = correct / total * 100.0
    
    return per_class_acc


def compute_top_k_accuracy(
    logits: Tensor,
    targets: Tensor,
    k: int = 5,
) -> float:
    _, top_k_preds = logits.topk(k, dim=1)
    correct = top_k_preds.eq(targets.unsqueeze(1)).any(dim=1)
    return correct.float().mean().item() * 100.0
