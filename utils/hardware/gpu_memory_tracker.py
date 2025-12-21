from dataclasses import dataclass
from dataclasses import field
from datetime import datetime
from pathlib import Path
from typing import Optional
import json

import torch


@dataclass
class MemorySnapshot:
    timestamp: str
    allocated_mb: float
    reserved_mb: float
    max_allocated_mb: float
    max_reserved_mb: float
    free_mb: float
    total_mb: float
    utilization_percent: float
    
    def to_dict(self) -> dict: 
        return {
            "timestamp": self.timestamp,
            "allocated_mb":  self.allocated_mb,
            "reserved_mb": self. reserved_mb,
            "max_allocated_mb": self. max_allocated_mb,
            "max_reserved_mb":  self.max_reserved_mb,
            "free_mb":  self.free_mb,
            "total_mb": self. total_mb,
            "utilization_percent": self.utilization_percent,
        }


@dataclass
class GPUInfo:
    device_id: int
    name: str
    total_memory_mb: float
    compute_capability: tuple[int, int]
    multi_processor_count: int


def get_gpu_info(device_id: int = 0) -> Optional[GPUInfo]: 
    if not torch.cuda.is_available():
        return None
    
    if device_id >= torch.cuda.device_count():
        return None
    
    props = torch.cuda. get_device_properties(device_id)
    
    return GPUInfo(
        device_id=device_id,
        name=props. name,
        total_memory_mb=props.total_memory / (1024 * 1024),
        compute_capability=(props.major, props.minor),
        multi_processor_count=props.multi_processor_count,
    )


class GPUMemoryTracker:
    
    def __init__(
        self,
        device_id: int = 0,
        warning_threshold_percent: float = 85.0,
        critical_threshold_percent: float = 95.0,
    ):
        self._device_id = device_id
        self._warning_threshold = warning_threshold_percent
        self._critical_threshold = critical_threshold_percent
        
        self._snapshots: list[MemorySnapshot] = []
        self._is_available = torch.cuda.is_available()
        
        if self._is_available:
            torch.cuda.set_device(device_id)
    
    def take_snapshot(self, label: Optional[str] = None) -> Optional[MemorySnapshot]:
        if not self._is_available:
            return None
        
        allocated = torch.cuda.memory_allocated(self._device_id) / (1024 * 1024)
        reserved = torch.cuda.memory_reserved(self._device_id) / (1024 * 1024)
        max_allocated = torch. cuda.max_memory_allocated(self._device_id) / (1024 * 1024)
        max_reserved = torch.cuda.max_memory_reserved(self._device_id) / (1024 * 1024)
        
        total = torch.cuda. get_device_properties(self._device_id).total_memory / (1024 * 1024)
        free = total - reserved
        utilization = (reserved / total) * 100 if total > 0 else 0
        
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        if label:
            timestamp = f"{timestamp} [{label}]"
        
        snapshot = MemorySnapshot(
            timestamp=timestamp,
            allocated_mb=allocated,
            reserved_mb=reserved,
            max_allocated_mb=max_allocated,
            max_reserved_mb=max_reserved,
            free_mb=free,
            total_mb=total,
            utilization_percent=utilization,
        )
        
        self._snapshots. append(snapshot)
        
        return snapshot
    
    def get_current_usage(self) -> tuple[float, float]: 
        if not self._is_available:
            return 0.0, 0.0
        
        allocated = torch.cuda. memory_allocated(self._device_id) / (1024 * 1024)
        reserved = torch.cuda.memory_reserved(self._device_id) / (1024 * 1024)
        
        return allocated, reserved
    
    def get_utilization_percent(self) -> float:
        if not self._is_available:
            return 0.0
        
        reserved = torch.cuda. memory_reserved(self._device_id)
        total = torch.cuda.get_device_properties(self._device_id).total_memory
        
        return (reserved / total) * 100 if total > 0 else 0.0
    
    def check_memory_status(self) -> tuple[str, float]: 
        utilization = self.get_utilization_percent()
        
        if utilization >= self._critical_threshold:
            status = "CRITICAL"
        elif utilization >= self._warning_threshold:
            status = "WARNING"
        else:
            status = "OK"
        
        return status, utilization
    
    def reset_peak_stats(self) -> None:
        if self._is_available:
            torch.cuda.reset_peak_memory_stats(self._device_id)
    
    def empty_cache(self) -> None:
        if self._is_available:
            torch.cuda.empty_cache()
    
    def get_peak_memory(self) -> float:
        if not self._is_available:
            return 0.0
        
        return torch. cuda.max_memory_allocated(self._device_id) / (1024 * 1024)
    
    def estimate_batch_size(
        self,
        sample_memory_mb: float,
        model_memory_mb:  float,
        safety_factor: float = 0.8,
    ) -> int:
        if not self._is_available:
            return 1
        
        total = torch.cuda.get_device_properties(self._device_id).total_memory / (1024 * 1024)
        
        available = (total * safety_factor) - model_memory_mb
        
        batch_size = int(available / sample_memory_mb) if sample_memory_mb > 0 else 1
        
        return max(1, batch_size)
    
    def save_snapshots(self, filepath: Path) -> None:
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        data = {
            "device_id": self._device_id,
            "snapshots": [s.to_dict() for s in self._snapshots],
        }
        
        with open(filepath, "w") as f:
            json. dump(data, f, indent=2)
    
    def get_summary(self) -> dict:
        if not self._snapshots:
            return {}
        
        allocated_values = [s.allocated_mb for s in self._snapshots]
        utilization_values = [s.utilization_percent for s in self._snapshots]
        
        return {
            "num_snapshots": len(self._snapshots),
            "allocated_mb_min": min(allocated_values),
            "allocated_mb_max": max(allocated_values),
            "allocated_mb_avg": sum(allocated_values) / len(allocated_values),
            "utilization_min": min(utilization_values),
            "utilization_max": max(utilization_values),
            "utilization_avg": sum(utilization_values) / len(utilization_values),
            "peak_memory_mb":  self.get_peak_memory(),
        }
    
    def clear_snapshots(self) -> None:
        self._snapshots. clear()