import copy

import torch
import torch.nn as nn
from torch import Tensor


class EMATeacher(nn. Module):
    
    def __init__(self, student:  nn.Module, decay: float = 0.999):
        super().__init__()
        
        self._decay = decay
        self._teacher = self._create_teacher_copy(student)
        self._update_count = 0
    
    def _create_teacher_copy(self, student:  nn.Module) -> nn.Module:
        teacher = copy.deepcopy(student)
        
        for param in teacher.parameters():
            param.requires_grad = False
        
        return teacher
    
    @torch.no_grad()
    def update(self, student: nn.Module) -> None:
        student_params = dict(student.named_parameters())
        teacher_params = dict(self._teacher.named_parameters())
        
        for name, teacher_param in teacher_params.items():
            if name in student_params:
                student_param = student_params[name]
                
                teacher_param.data.mul_(self._decay)
                teacher_param.data.add_(student_param.data, alpha=1 - self._decay)
        
        student_buffers = dict(student.named_buffers())
        teacher_buffers = dict(self._teacher.named_buffers())
        
        for name, teacher_buffer in teacher_buffers.items():
            if name in student_buffers:
                teacher_buffer.data.copy_(student_buffers[name].data)
        
        self._update_count += 1
    
    def synchronize_architecture(self, student: nn.Module) -> None:
        self._teacher = self._create_teacher_copy(student)
        self._update_count = 0
    
    def forward(self, x: Tensor) -> Tensor:
        return self._teacher(x)
    
    @property
    def update_count(self) -> int:
        return self._update_count