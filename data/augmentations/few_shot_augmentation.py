import torch.nn as nn
from torch import Tensor
from torchvision import transforms as T


class FewShotAugmentation(nn.Module):
    
    def __init__(self, image_size: int, normalize:  bool = True):
        super().__init__()
        self._image_size = image_size
        self._normalize = normalize
        self._transform = self._build_transform()
    
    def _build_transform(self) -> T.Compose:
        transform_list = [
            T.Resize((self._image_size, self._image_size)),
        ]
        
        if self._normalize:
            transform_list.append(
                T.Normalize(
                    mean=(0.485, 0.456, 0.406),
                    std=(0.229, 0.224, 0.225),
                )
            )
        
        return T.Compose(transform_list)
    
    def forward(self, image:  Tensor) -> Tensor:
        return self._transform(image)