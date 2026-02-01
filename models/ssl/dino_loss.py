import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
import numpy as np
class DINOLoss(nn.Module):
    def __init__(
        self,
        out_dim: int,
        warmup_teacher_temp: float = 0.04,
        teacher_temp: float = 0.04,
        warmup_teacher_temp_epochs: int = 0,
        n_epochs: int = 100,
        student_temp: float = 0.1,
        center_momentum: float = 0.9,
    ):
        super().__init__()
        self.student_temp = student_temp
        self.center_momentum = center_momentum
        self.register_buffer("center", torch.zeros(1, out_dim))
        self.teacher_temp_schedule = np.concatenate((
            np.linspace(warmup_teacher_temp,
                        teacher_temp, warmup_teacher_temp_epochs),
            np.ones(n_epochs - warmup_teacher_temp_epochs) * teacher_temp
        ))
    def forward(self, student_output: list[Tensor], teacher_output: list[Tensor], epoch: int) -> Tensor:
        student_out = torch.cat([x / self.student_temp for x in student_output])
        student_out = student_out.chunk(len(student_output))
        temp = self.teacher_temp_schedule[epoch]
        teacher_out = F.softmax((torch.cat(teacher_output) - self.center) / temp, dim=-1)
        teacher_out = teacher_out.detach().chunk(len(teacher_output))
        total_loss = 0
        n_loss_terms = 0
        for iq, q in enumerate(teacher_out):
            for v in range(len(student_out)):
                if v == iq:
                    continue
                loss = torch.sum(-q * F.log_softmax(student_out[v], dim=-1), dim=-1)
                total_loss += loss.mean()
                n_loss_terms += 1
        total_loss /= n_loss_terms
        self.update_center(torch.cat(teacher_output))
        return total_loss
    @torch.no_grad()
    def update_center(self, teacher_output: Tensor) -> None:
        batch_center = torch.sum(teacher_output, dim=0, keepdim=True)
        batch_center = batch_center / len(teacher_output)
        self.center = self.center * self.center_momentum + batch_center * (1 - self.center_momentum)
