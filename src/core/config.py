from __future__ import annotations

import yaml
from dataclasses import dataclass, asdict
from typing import Any


@dataclass
class ConfigBase:
    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_yaml(cls, path: str) -> "ConfigBase":
        with open(path) as f:
            raw = yaml.safe_load(f)
        return cls(**raw)

    def validate(self) -> list[str]:
        return []