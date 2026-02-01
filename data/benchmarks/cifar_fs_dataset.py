from pathlib import Path
from typing import Literal
from torch import Tensor
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
SplitType = Literal["train", "val", "test"]
class CIFARFSDataset(Dataset):
    MEAN = (0.5071, 0.4867, 0.4408)
    STD = (0.2675, 0.2565, 0.2761)
    def __init__(
        self,
        root_dir: Path,
        split: SplitType,
        image_size: int = 84,
        augment: bool = False,
    ):
        self._root_dir = root_dir
        self._split = split
        self._image_size = image_size
        self._augment = augment
        self._samples: list[tuple[Path, int]] = []
        self._class_to_idx: dict[str, int] = {}
        self._idx_to_class: dict[int, str] = {}
        self._transform = self._build_transform()
        self._load_dataset()
    def _build_transform(self) -> transforms.Compose:
        transform_list = []
        if self._augment:
            transform_list.extend([
                transforms.RandomResizedCrop(
                    self._image_size,
                    scale=(0.8, 1.0),
                ),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.ColorJitter(
                    brightness=0.4,
                    contrast=0.4,
                    saturation=0.4,
                    hue=0.1,
                ),
            ])
        else:
            transform_list.append(
                transforms.Resize((self._image_size, self._image_size))
            )
        transform_list.extend([
            transforms.ToTensor(),
            transforms.Normalize(mean=self.MEAN, std=self.STD),
        ])
        return transforms.Compose(transform_list)
    def _load_dataset(self) -> None:
        split_dir = self._root_dir / self._split
        if not split_dir.exists():
            raise FileNotFoundError(f"Directory not found: {split_dir}")
        class_dirs = sorted([d for d in split_dir.iterdir() if d.is_dir()])
        for class_idx, class_dir in enumerate(class_dirs):
            class_name = class_dir.name
            self._class_to_idx[class_name] = class_idx
            self._idx_to_class[class_idx] = class_name
            valid_extensions = {".jpg", ".jpeg", ".png", ".JPEG", ". JPG", ".PNG"}
            for image_path in class_dir.iterdir():
                if image_path.suffix in valid_extensions:
                    self._samples.append((image_path, class_idx))
    def __len__(self) -> int:
        return len(self._samples)
    def __getitem__(self, index: int) -> tuple[Tensor, int]: 
        image_path, label = self._samples[index]
        image = Image.open(image_path).convert("RGB")
        image_tensor = self._transform(image)
        return image_tensor, label
    @property
    def num_classes(self) -> int:
        return len(self._class_to_idx)
    @property
    def class_to_idx(self) -> dict[str, int]:
        return self._class_to_idx.copy()
    def get_samples_by_class(self, class_idx:  int) -> list[int]:
        return [
            idx
            for idx, (_, label) in enumerate(self._samples)
            if label == class_idx
        ]
