from typing import Tuple
import torch
import torch.nn as nn
from torch import Tensor
from torchvision import transforms as T
from configs.ssl_config import AugmentationConfig
class Solarize(object):
    def __init__(self, threshold=128):
        self.threshold = threshold
    def __call__(self, tensor):
        return torch.where(tensor < self.threshold/255.0, tensor, 1.0 - tensor)
class SSLAugmentation(nn.Module):
    def __init__(self, config: AugmentationConfig, image_size: int):
        super().__init__()
        self._config = config
        self._image_size = image_size
        self._global_crops_scale = (0.4, 1.0) 
        self._local_crops_scale = (0.05, 0.4) 
        self._local_crops_number = 6 
        self._global_transfo1 = self._build_transform(is_global=True)
        self._global_transfo2 = self._build_transform(is_global=True)
        self._local_transfo = self._build_transform(is_global=False)
    def _build_transform(self, is_global: bool) -> T.Compose:
        if is_global:
            crop_scale = self._global_crops_scale
            size = self._image_size
        else:
            crop_scale = self._local_crops_scale
            size = 96 
            if self._image_size <= 84:
                 size = 32 
        aug_list = [
            T.RandomResizedCrop(
                size,
                scale=crop_scale,
                interpolation=T.InterpolationMode.BICUBIC,
            ),
            T.RandomHorizontalFlip(p=self._config.horizontal_flip_prob),
            T.RandomApply([
                T.ColorJitter(
                    brightness=self._config.color_jitter_strength,
                    contrast=self._config.color_jitter_strength,
                    saturation=self._config.color_jitter_strength,
                    hue=self._config.color_jitter_strength / 4,
                )
            ], p=0.8),
            T.RandomGrayscale(p=self._config.grayscale_prob),
            T.GaussianBlur(
                kernel_size=self._config.gaussian_blur_kernel_size,
                sigma=(0.1, 2.0),
            ),
            T.RandomApply([Solarize(128)], p=0.0 if not is_global else 0.2), 
        ]
        return T.Compose(aug_list)
    def forward(self, image: Tensor) -> list[Tensor]: 
        crops = []
        crops.append(self._global_transfo1(image))
        crops.append(self._global_transfo2(image))
        for _ in range(self._local_crops_number):
            crops.append(self._local_transfo(image))
        return crops
