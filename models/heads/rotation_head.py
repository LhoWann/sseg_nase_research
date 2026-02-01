from torch import nn
from torch import Tensor
from configs.ssl_config import ProjectionConfig
class RotationHead(nn.Module):
    def __init__(self, input_dim: int, num_classes: int = 4):
        super().__init__()
        self.head = nn.Linear(input_dim, num_classes)
    def forward(self, x: Tensor) -> Tensor:
        return self.head(x)
