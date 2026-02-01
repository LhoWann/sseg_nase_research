import torch.nn as nn
from torch import Tensor
from configs.ssl_config import ProjectionConfig
class ProjectionHead(nn.Module):
    def __init__(self, input_dim: int, config: ProjectionConfig):
        super().__init__()
        self._config = config
        self._input_dim = input_dim
        self.layers = self._build_layers()
    def _build_layers(self) -> nn.Sequential:
        layers = []
        current_dim = self._input_dim
        for i in range(self._config.num_layers - 1):
            layers.append(nn.Linear(current_dim, self._config.hidden_dim))
            if self._config.use_batch_norm:
                layers.append(nn.BatchNorm1d(self._config.hidden_dim))
            layers.append(nn.ReLU(inplace=True))
            if hasattr(self._config, "dropout") and self._config.dropout > 0:
                layers.append(nn.Dropout(p=self._config.dropout))
            current_dim = self._config.hidden_dim
        layers.append(nn.Linear(current_dim, self._config.output_dim))
        if hasattr(self._config, "dropout") and self._config.dropout > 0:
            layers.append(nn.Dropout(p=self._config.dropout))
        return nn.Sequential(*layers)
    def forward(self, x: Tensor) -> Tensor:
        return self.layers(x)
