from dataclasses import dataclass
from enum import Enum
from typing import Optional

import torch
from torch import Tensor


class ShapeType(Enum):
    CIRCLE = "circle"
    SQUARE = "square"
    TRIANGLE = "triangle"
    DIAMOND = "diamond"


@dataclass(frozen=True)
class GenerationConfig:
    image_size: int
    seed: Optional[int] = None


class SyntheticGenerator:
    
    def __init__(self, config: GenerationConfig):
        self._config = config
        self._generator = self._create_generator()
    
    def _create_generator(self) -> torch.Generator:
        generator = torch.Generator()
        if self._config.seed is not None:
            generator.manual_seed(self._config.seed)
        return generator
    
    def generate_basic_shape(self) -> Tensor:
        image = torch.zeros(3, self._config.image_size, self._config.image_size)
        
        color = torch.rand(3, generator=self._generator)
        shape_type = ShapeType(
            list(ShapeType)[
                torch.randint(0, len(ShapeType), (1,), generator=self._generator).item()
            ]
        )
        
        center_y = self._config.image_size // 2
        center_x = self._config.image_size // 2
        radius = torch.randint(
            self._config.image_size // 8,
            self._config.image_size // 3,
            (1,),
            generator=self._generator
        ).item()
        
        y_coords = torch.arange(self._config.image_size).view(-1, 1)
        x_coords = torch. arange(self._config.image_size).view(1, -1)
        
        mask = self._create_shape_mask(
            shape_type, y_coords, x_coords, center_y, center_x, radius
        )
        
        for c in range(3):
            image[c][mask] = color[c]
        
        return image
    
    def _create_shape_mask(
        self,
        shape_type: ShapeType,
        y_coords:  Tensor,
        x_coords: Tensor,
        center_y: int,
        center_x: int,
        radius:  int,
    ) -> Tensor:
        if shape_type == ShapeType. CIRCLE:
            dist_sq = (y_coords - center_y) ** 2 + (x_coords - center_x) ** 2
            return dist_sq <= radius ** 2
        
        elif shape_type == ShapeType.SQUARE:
            return (torch.abs(y_coords - center_y) <= radius) & (
                torch. abs(x_coords - center_x) <= radius
            )
        
        elif shape_type == ShapeType.DIAMOND:
            manhattan_dist = torch.abs(y_coords - center_y) + torch.abs(
                x_coords - center_x
            )
            return manhattan_dist <= radius
        
        else:
            y_from_center = y_coords - center_y
            x_from_center = x_coords - center_x
            
            apex_y = -radius
            base_y = radius
            
            in_y_bounds = (y_from_center >= apex_y) & (y_from_center <= base_y)
            
            slope = (y_from_center - apex_y) / (base_y - apex_y + 1e-6)
            max_x_offset = slope * radius
            
            in_x_bounds = torch.abs(x_from_center) <= max_x_offset
            
            return in_y_bounds & in_x_bounds
    
    def generate_texture(self) -> Tensor:
        image = torch.zeros(3, self._config.image_size, self._config.image_size)
        
        base_freq = torch.rand(1, generator=self._generator).item() * 10 + 5
        
        y_coords = torch.arange(self._config.image_size, dtype=torch.float32).view(-1, 1)
        x_coords = torch.arange(self._config.image_size, dtype=torch.float32).view(1, -1)
        
        for c in range(3):
            phase = torch.rand(1, generator=self._generator).item() * 2 * 3.14159
            freq_variation = torch.rand(1, generator=self._generator).item() * 2
            
            frequency = base_freq + freq_variation
            normalized_coords = (y_coords + x_coords) / self._config.image_size
            
            pattern = torch.sin(normalized_coords * frequency * 2 * 3.14159 + phase)
            
            noise = torch.rand(
                self._config.image_size,
                self._config.image_size,
                generator=self._generator,
            ) * 0.2
            
            image[c] = (pattern + 1) / 2 + noise
        
        return torch.clamp(image, 0, 1)
    
    def generate_complex_object(self) -> Tensor:
        background_color = (
            torch.rand(3, generator=self._generator) * 0.3
        ).view(3, 1, 1)
        
        image = background_color. expand(
            3, self._config.image_size, self._config.image_size
        ).clone()
        
        num_shapes = torch.randint(
            2, 5, (1,), generator=self._generator
        ).item()
        
        for _ in range(num_shapes):
            shape = self. generate_basic_shape()
            
            offset_range = self._config.image_size // 5
            offset_y = torch.randint(
                -offset_range, offset_range + 1, (1,), generator=self._generator
            ).item()
            offset_x = torch.randint(
                -offset_range, offset_range + 1, (1,), generator=self._generator
            ).item()
            
            if offset_y != 0 or offset_x != 0:
                shape = torch.roll(shape, shifts=(offset_y, offset_x), dims=(1, 2))
            
            mask = shape. sum(dim=0) > 0
            image[: , mask] = shape[: , mask]
        
        return torch.clamp(image, 0, 1)
    
    def generate_adversarial(self) -> Tensor:
        base_image = self.generate_complex_object()
        
        perturbation = (
            torch.randn(
                3,
                self._config.image_size,
                self._config.image_size,
                generator=self._generator,
            )
            * 0.1
        )
        
        adversarial_image = base_image + perturbation
        
        return torch.clamp(adversarial_image, 0, 1)