from dataclasses import dataclass
from datetime import datetime
from enum import IntEnum
from pathlib import Path
from typing import Optional
import sys


class LogLevel(IntEnum):
    DEBUG = 10
    INFO = 20
    WARNING = 30
    ERROR = 40
    CRITICAL = 50


@dataclass
class LogRecord:
    timestamp:  str
    level:  LogLevel
    module: str
    message: str
    
    def format(self, include_timestamp: bool = True) -> str:
        level_name = self. level.name
        
        if include_timestamp:
            return f"[{self.timestamp}] [{level_name: 8s}] [{self.module}] {self.message}"
        return f"[{level_name:8s}] [{self. module}] {self.message}"


class CustomLogger:
    
    _instances:  dict[str, "CustomLogger"] = {}
    
    def __init__(
        self,
        name: str,
        level: LogLevel = LogLevel.INFO,
        log_file: Optional[Path] = None,
        console_output: bool = True,
    ):
        self._name = name
        self._level = level
        self._log_file = log_file
        self._console_output = console_output
        
        if log_file:
            log_file.parent.mkdir(parents=True, exist_ok=True)
            self._file_handle = open(log_file, "a")
        else: 
            self._file_handle = None
    
    @classmethod
    def get_instance(
        cls,
        name: str,
        level: LogLevel = LogLevel.INFO,
        log_file: Optional[Path] = None,
    ) -> "CustomLogger":
        if name not in cls._instances:
            cls._instances[name] = cls(name, level, log_file)
        return cls._instances[name]
    
    def _create_record(self, level: LogLevel, message: str) -> LogRecord:
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        return LogRecord(
            timestamp=timestamp,
            level=level,
            module=self._name,
            message=message,
        )
    
    def _write(self, record: LogRecord) -> None:
        if record.level < self._level:
            return
        
        formatted = record.format()
        
        if self._console_output:
            stream = sys.stderr if record.level >= LogLevel.ERROR else sys.stdout
            print(formatted, file=stream)
        
        if self._file_handle:
            self._file_handle.write(formatted + "\n")
            self._file_handle.flush()
    
    def debug(self, message: str) -> None:
        record = self._create_record(LogLevel. DEBUG, message)
        self._write(record)
    
    def info(self, message: str) -> None:
        record = self._create_record(LogLevel.INFO, message)
        self._write(record)
    
    def warning(self, message: str) -> None:
        record = self._create_record(LogLevel.WARNING, message)
        self._write(record)
    
    def error(self, message:  str) -> None:
        record = self._create_record(LogLevel.ERROR, message)
        self._write(record)
    
    def critical(self, message: str) -> None:
        record = self._create_record(LogLevel. CRITICAL, message)
        self._write(record)
    
    def log_architecture(
        self,
        num_blocks: int,
        num_params: int,
        feature_dim: int,
    ) -> None:
        message = (
            f"Architecture: blocks={num_blocks}, "
            f"params={num_params: ,}, "
            f"feature_dim={feature_dim}"
        )
        self.info(message)
    
    def log_training_step(
        self,
        epoch: int,
        step: int,
        total_steps: int,
        loss: float,
        lr: float,
    ) -> None:
        message = (
            f"Epoch {epoch} [{step}/{total_steps}] "
            f"loss={loss:.4f} lr={lr:.2e}"
        )
        self.info(message)
    
    def log_epoch_summary(
        self,
        epoch: int,
        ssl_loss: float,
        distill_loss: float,
        curriculum_level: int,
    ) -> None:
        message = (
            f"Epoch {epoch} complete:  "
            f"ssl_loss={ssl_loss:.4f}, "
            f"distill_loss={distill_loss:.4f}, "
            f"level={curriculum_level}"
        )
        self.info(message)
    
    def log_evolution_event(
        self,
        mutation_type: str,
        target_layer: Optional[int],
        params_before: int,
        params_after: int,
    ) -> None:
        delta = params_after - params_before
        message = (
            f"Evolution:  {mutation_type} "
            f"target={target_layer} "
            f"params={params_before: ,}->{params_after:,} "
            f"(delta={delta: +,})"
        )
        self.info(message)
    
    def log_evaluation_result(
        self,
        num_shots: int,
        mean_accuracy: float,
        margin:  float,
    ) -> None:
        message = (
            f"Evaluation: {num_shots}-shot "
            f"accuracy={mean_accuracy:. 2f}%±{margin:.2f}%"
        )
        self.info(message)
    
    def set_level(self, level: LogLevel) -> None:
        self._level = level
    
    def close(self) -> None:
        if self._file_handle:
            self._file_handle.close()
            self._file_handle = None


def get_logger(
    name: str,
    level: LogLevel = LogLevel. INFO,
    log_file: Optional[Path] = None,
) -> CustomLogger: 
    return CustomLogger. get_instance(name, level, log_file)