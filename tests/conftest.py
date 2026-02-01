from pathlib import Path
import random
import pytest
import torch
import numpy as np
def set_seed(seed:  int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
@pytest.fixture(autouse=True)
def set_random_seed() -> None:
    set_seed(42)
@pytest.fixture
def device() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"
@pytest.fixture
def sample_image_batch() -> torch.Tensor:
    return torch.randn(4, 3, 84, 84)
@pytest.fixture
def sample_labels() -> torch.Tensor:
    return torch.tensor([0, 0, 1, 1])
@pytest.fixture
def tmp_output_dir(tmp_path: Path) -> Path:
    output_dir = tmp_path / "test_outputs"
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir
