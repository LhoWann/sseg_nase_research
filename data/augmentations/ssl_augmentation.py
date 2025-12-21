from typing import Tuple

import torch
import torch.nn as nn
from torch import Tensor
from torchvision import transforms as T

from configs.ssl_config import AugmentationConfig


class SSLAugmentation(nn.Module):
    
    def __init__(self, config: AugmentationConfig, image_size: int):
        super().__init__()
        self._config = config
        self._image_size = image_size
        self._transform = self._build_transform()
    
    def _build_transform(self) -> T.Compose:
        return T.Compose([
            T.RandomResizedCrop(
                self._image_size,
                scale=(self._config.crop_scale_min, self._config.crop_scale_max),
            ),
            T.RandomHorizontalFlip(p=self._config.horizontal_flip_prob),
            T.RandomApply(
                [
                    T. ColorJitter(
                        brightness=self._config.color_jitter_strength,
                        contrast=self._config.color_jitter_strength,
                        saturation=self._config.color_jitter_strength,
                        hue=self._config.color_jitter_strength / 4,
                    )
                ],
                p=0.8,
            ),
            T.RandomGrayscale(p=self._config.grayscale_prob),
            T.GaussianBlur(
                kernel_size=self._config.gaussian_blur_kernel_size,
                sigma=(0.1, 2.0),
            ),
        ])
    
    def forward(self, image: Tensor) -> Tuple[Tensor, Tensor]: 
        view_1 = self._transform(image)
        view_2 = self._transform(image)
        return view_1, view_2