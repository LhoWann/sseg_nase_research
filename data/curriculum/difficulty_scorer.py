from dataclasses import dataclass
import torch
import torch.nn.functional as F
from torch import Tensor
@dataclass(frozen=True)
class DifficultyComponents:
    edge_density: float
    color_variance: float
    spatial_frequency: float
    def aggregate(self, weights: tuple[float, float, float]) -> float:
        if abs(sum(weights) - 1.0) > 1e-6:
            raise ValueError("weights must sum to 1.0")
        return (
            weights[0] * self.edge_density
            + weights[1] * self.color_variance
            + weights[2] * self.spatial_frequency
        )
class DifficultyScorer: 
    def __init__(self, device: str = "cpu"):
        self._device = device
        self._sobel_x = self._create_sobel_kernel("x").to(device)
        self._sobel_y = self._create_sobel_kernel("y").to(device)
    def _create_sobel_kernel(self, direction: str) -> Tensor:
        if direction == "x":
            kernel = torch.tensor(
                [[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]]
            )
        else:
            kernel = torch.tensor(
                [[-1.0, -2.0, -1.0], [0.0, 0.0, 0.0], [1.0, 2.0, 1.0]]
            )
        return kernel.view(1, 1, 3, 3)
    def compute_edge_density(self, image:  Tensor) -> float:
        if image.dim() == 3:
            image = image.unsqueeze(0)
        grayscale = image.mean(dim=1, keepdim=True)
        edges_x = F.conv2d(grayscale, self._sobel_x, padding=1)
        edges_y = F.conv2d(grayscale, self._sobel_y, padding=1)
        edge_magnitude = torch.sqrt(edges_x**2 + edges_y**2)
        return edge_magnitude.mean().item()
    def compute_color_variance(self, image: Tensor) -> float:
        if image.dim() == 3:
            image = image.unsqueeze(0)
        per_channel_var = image.var(dim=(2, 3))
        return per_channel_var.mean().item()
    def compute_spatial_frequency(self, image: Tensor) -> float:
        if image.dim() == 3:
            image = image.unsqueeze(0)
        grayscale = image.mean(dim=1)
        fft_result = torch.fft.fft2(grayscale)
        magnitude = torch.abs(fft_result)
        h, w = magnitude.shape[-2:]
        center_h, center_w = h // 2, w // 2
        y_coords = torch.arange(h, device=self._device) - center_h
        x_coords = torch.arange(w, device=self._device) - center_w
        y_grid, x_grid = torch.meshgrid(y_coords, x_coords, indexing="ij")
        frequency_weights = torch.sqrt(y_grid**2 + x_grid**2)
        weighted_magnitude = magnitude * frequency_weights
        mean_magnitude = magnitude.mean().item()
        if mean_magnitude < 1e-8:
            return 0.0
        return weighted_magnitude.mean().item() / mean_magnitude
    def score(self, image: Tensor) -> DifficultyComponents: 
        edge_density = self.compute_edge_density(image)
        color_variance = self.compute_color_variance(image)
        spatial_frequency = self.compute_spatial_frequency(image)
        max_edge = 10.0
        max_color = 1.0
        max_freq = 50.0
        return DifficultyComponents(
            edge_density=min(edge_density / max_edge, 1.0),
            color_variance=min(color_variance / max_color, 1.0),
            spatial_frequency=min(spatial_frequency / max_freq, 1.0),
        )
