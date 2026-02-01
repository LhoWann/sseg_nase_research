import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
class CosineClassifier(nn.Module):
    def __init__(self, in_features: int, num_classes: int, scale: float = 10.0, learn_scale: bool = True):
        super().__init__()
        self.in_features = in_features
        self.num_classes = num_classes
        self.weight = nn.Parameter(torch.Tensor(num_classes, in_features))
        nn.init.normal_(self.weight, std=0.01)
        if learn_scale:
            self.scale = nn.Parameter(torch.tensor(scale))
        else:
            self.register_buffer('scale', torch.tensor(scale))
    def forward(self, x: Tensor) -> Tensor:
        x_norm = F.normalize(x, p=2, dim=1)
        w_norm = F.normalize(self.weight, p=2, dim=1)
        cosine_sim = F.linear(x_norm, w_norm)
        logits = self.scale * cosine_sim
        return logits
