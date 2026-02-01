from pathlib import Path
from typing import Any
from typing import Optional
from typing import Type
from typing import TypeVar
import json
import yaml
T = TypeVar("T")
class ConfigLoader:
    SUPPORTED_EXTENSIONS = {".yaml", ".yml", ".json"}
    @classmethod
    def load(cls, filepath: Path) -> dict[str, Any]: 
        if not filepath.exists():
            raise FileNotFoundError(f"Config file not found:  {filepath}")
        suffix = filepath.suffix.lower()
        if suffix not in cls.SUPPORTED_EXTENSIONS: 
            raise ValueError(
                f"Unsupported file extension: {suffix}. Supported: {cls.SUPPORTED_EXTENSIONS}"
            )
        with open(filepath, "r") as f:
            if suffix in {".yaml", ".yml"}:
                config = yaml.safe_load(f)
            else:
                config = json.load(f)
        return config
    @classmethod
    def save(cls, config:  dict[str, Any], filepath: Path) -> None:
        filepath.parent.mkdir(parents=True, exist_ok=True)
        suffix = filepath.suffix.lower()
        if suffix not in cls.SUPPORTED_EXTENSIONS: 
            raise ValueError(
                f"Unsupported file extension: {suffix}. Supported: {cls.SUPPORTED_EXTENSIONS}"
            )
        with open(filepath, "w") as f:
            if suffix in {".yaml", ".yml"}:
                yaml.dump(config, f, default_flow_style=False, indent=2)
            else:
                json.dump(config, f, indent=2)
    @classmethod
    def load_with_overrides(
        cls,
        filepath: Path,
        overrides: Optional[dict[str, Any]] = None,
    ) -> dict[str, Any]:
        config = cls.load(filepath)
        if overrides: 
            config = cls._deep_merge(config, overrides)
        return config
    @classmethod
    def _deep_merge(
        cls,
        base: dict[str, Any],
        override: dict[str, Any],
    ) -> dict[str, Any]:
        result = base.copy()
        for key, value in override.items():
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                result[key] = cls._deep_merge(result[key], value)
            else:
                result[key] = value
        return result
    @classmethod
    def validate_required_keys(
        cls,
        config: dict[str, Any],
        required_keys: list[str],
    ) -> None:
        missing = []
        for key in required_keys:
            parts = key.split(".")
            current = config
            for part in parts: 
                if isinstance(current, dict) and part in current: 
                    current = current[part]
                else:
                    missing.append(key)
                    break
        if missing: 
            raise KeyError(f"Missing required config keys: {missing}")
    @classmethod
    def get_nested(
        cls,
        config: dict[str, Any],
        key_path: str,
        default: Optional[Any] = None,
    ) -> Any:
        parts = key_path.split(".")
        current = config
        for part in parts:
            if isinstance(current, dict) and part in current:
                current = current[part]
            else:
                return default
        return current
    @classmethod
    def set_nested(
        cls,
        config: dict[str, Any],
        key_path: str,
        value: Any,
    ) -> dict[str, Any]: 
        parts = key_path.split(".")
        current = config
        for part in parts[:-1]:
            if part not in current: 
                current[part] = {}
            current = current[part]
        current[parts[-1]] = value
        return config
