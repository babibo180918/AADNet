import os
from typing import Any, Union
import yaml
from pathlib import Path

class Config:
    """Configuration class."""

    def __init__(self, config: dict[str, Any]):
        self.config = config

    def get(self, key: Union[str, tuple[str, ...]], **kwargs) -> Any:
        """Get a configuration value.
        Args:
            key: The key to get.
            fallback: The fallback value if the key is not found.
                if fallback is not provided, an exception is raised on missing key.
        """
        if isinstance(key, tuple):
            value = self.config
            for k in key:
                if k not in value:
                    if "fallback" in kwargs:
                        return kwargs["fallback"]
                    msg = f"Key not found: {key}"
                    raise KeyError(msg)
                value = value[k]
            return value
        if key not in self.config:
            if "fallback" in kwargs:
                return kwargs["fallback"]
            msg = f"Key not found: {key}"
            raise KeyError(msg)
        return self.config[key]

    def set(self, key, value):
        """Set a configuration value."""
        self.config[key] = value
        return self.config

    def __getitem__(self, key):
        return self.get(key)

    def __setitem__(self, key, value):
        return self.set(key, value)

    def __repr__(self):
        return f"<Config {self.config}>"

    def _update(self, d: dict[str, Any], o: dict[str, Any]) -> dict[str, Any]:
        """Update the configuration."""
        for k, v in o.items():
            if isinstance(v, Mapping):
                d[k] = self._update(d.get(k, {}), v)
            else:
                d[k] = v
        return d

    def merge(self, other: "Config") -> "Config":
        """Merge two configurations."""
        if not isinstance(other, Config):
            msg = "Can only merge Config objects"
            raise ValueError(msg)
        self.config = self._update(self.config, other.config)

        return self

    @classmethod
    def load_config(__cls__, path: Union[Path, str, dict]) -> Any:
        """Load a configuration file."""
        if isinstance(path, dict):
            return __cls__(path)
        with open(path, encoding="utf8") as file:
            config = yaml.safe_load(file)
        return __cls__(config)