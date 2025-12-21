from typing import Literal

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class NTXentLoss(nn. Module):
    
    def __init__(self, temperature: float = 0.5, normalize: bool = True):
        super().__init__()
        
        self._temperature = temperature
        self._normalize = normalize
    
    def forward(self, z_i: Tensor, z_j:  Tensor) -> Tensor:
        batch_size = z_i.size(0)
        
        if self._normalize:
            z_i = F.normalize(z_i, dim=1)
            z_j = F.normalize(z_j, dim=1)
        
        representations = torch.cat([z_i, z_j], dim=0)
        
        similarity_matrix = F.cosine_similarity(
            representations.unsqueeze(1), representations.unsqueeze(0), dim=2
        )
        
        sim_ij = torch.diag(similarity_matrix, batch_size)
        sim_ji = torch.diag(similarity_matrix, -batch_size)
        positives = torch.cat([sim_ij, sim_ji], dim=0)
        
        mask = torch.eye(2 * batch_size, device=z_i.device, dtype=torch.bool)
        negatives = similarity_matrix[~mask].view(2 * batch_size, -1)
        
        logits = torch.cat([positives.unsqueeze(1), negatives], dim=1)
        logits = logits / self._temperature
        
        labels = torch.zeros(2 * batch_size, dtype=torch.long, device=z_i.device)
        
        loss = F.cross_entropy(logits, labels)
        
        return loss


class DistillationLoss(nn.Module):
    
    def __init__(
        self,
        loss_type: Literal["mse", "cosine", "kl"] = "mse",
        normalize: bool = True,
    ):
        super().__init__()
        
        self._loss_type = loss_type
        self._normalize = normalize
    
    def forward(self, student_features: Tensor, teacher_features:  Tensor) -> Tensor:
        if self._normalize:
            student_features = F.normalize(student_features, dim=1)
            teacher_features = F.normalize(teacher_features, dim=1)
        
        if self._loss_type == "mse":
            loss = F.mse_loss(student_features, teacher_features)
        
        elif self._loss_type == "cosine":
            loss = 1 - F.cosine_similarity(student_features, teacher_features, dim=1).mean()
        
        else: 
            student_log_probs = F.log_softmax(student_features, dim=1)
            teacher_probs = F.softmax(teacher_features, dim=1)
            loss = F.kl_div(student_log_probs, teacher_probs, reduction="batchmean")
        
        return loss


class CombinedSSLLoss(nn.Module):
    
    def __init__(
        self,
        temperature: float = 0.5,
        distillation_weight: float = 0.5,
        distillation_type: Literal["mse", "cosine", "kl"] = "mse",
    ):
        super().__init__()
        
        self.contrastive_loss = NTXentLoss(temperature=temperature)
        self.distillation_loss = DistillationLoss(loss_type=distillation_type)
        self._distillation_weight = distillation_weight
    
    def forward(
        self,
        z_i: Tensor,
        z_j: Tensor,
        student_features: Tensor,
        teacher_features: Tensor,
    ) -> tuple[Tensor, dict[str, float]]:
        contrastive = self.contrastive_loss(z_i, z_j)
        distillation = self.distillation_loss(student_features, teacher_features)
        
        total_loss = contrastive + self._distillation_weight * distillation
        
        loss_components = {
            "contrastive": contrastive.item(),
            "distillation":  distillation.item(),
            "total": total_loss.item(),
        }
        
        return total_loss, loss_components