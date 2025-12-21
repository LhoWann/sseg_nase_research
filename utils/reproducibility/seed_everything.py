from dataclasses import dataclass
from typing import Optional
import os
import random

import numpy as np
import torch


@dataclass
class RandomState:
    python_state: tuple
    numpy_state:  dict
    torch_state: torch.Tensor
    cuda_state: Optional[list[torch.Tensor]]
    
    def to_dict(self) -> dict: 
        return {
            "python_state": self.python_state,
            "numpy_state": {
                "keys": self.numpy_state["keys"],
                "pos": self.numpy_state["pos"],
            },
            "torch_state": self. torch_state.tolist(),
            "cuda_state": (
                [s.tolist() for s in self.cuda_state]
                if self. cuda_state
                else None
            ),
        }


def seed_everything(seed: int, deterministic: bool = True) -> None:
    random.seed(seed)
    
    os.environ["PYTHONHASHSEED"] = str(seed)
    
    np.random.seed(seed)
    
    torch.manual_seed(seed)
    
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    
    if deterministic:
        torch. backends.cudnn. deterministic = True
        torch.backends.cudnn.benchmark = False
        
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
        
        try:
            torch.use_deterministic_algorithms(True)
        except Exception:
            pass


def get_random_state() -> RandomState: 
    python_state = random. getstate()
    
    numpy_state = np.random.get_state()
    numpy_state_dict = {
        "keys": numpy_state[1]. tolist(),
        "pos": numpy_state[2],
    }
    
    torch_state = torch.get_rng_state()
    
    cuda_state = None
    if torch. cuda.is_available():
        cuda_state = [
            torch.cuda.get_rng_state(device=i)
            for i in range(torch. cuda.device_count())
        ]
    
    return RandomState(
        python_state=python_state,
        numpy_state=numpy_state_dict,
        torch_state=torch_state,
        cuda_state=cuda_state,
    )


def set_random_state(state: RandomState) -> None:
    random.setstate(state.python_state)
    
    np_state = (
        "MT19937",
        np.array(state.numpy_state["keys"], dtype=np.uint32),
        state.numpy_state["pos"],
        0,
        0. 0,
    )
    np.random.set_state(np_state)
    
    torch. set_rng_state(state.torch_state)
    
    if state. cuda_state and torch.cuda.is_available():
        for i, cuda_s in enumerate(state. cuda_state):
            if i < torch.cuda.device_count():
                torch.cuda.set_rng_state(cuda_s, device=i)


def get_worker_init_fn(seed: int):
    def worker_init_fn(worker_id:  int) -> None:
        worker_seed = seed + worker_id
        random.seed(worker_seed)
        np.random.seed(worker_seed)
        torch.manual_seed(worker_seed)
    
    return worker_init_fn