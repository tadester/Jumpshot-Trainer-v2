from __future__ import annotations

from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any
import json


@dataclass(slots=True)
class AthleteProfile:
    athlete_id: str
    name: str
    handedness: str
    height_m: float
    wingspan_m: float
    standing_reach_m: float
    age: int | None = None
    skill_level: str | None = None
    capture_fps: int | None = None
    resolution: str | None = None
    notes: str | None = None
    source_tier: str | None = None

    @classmethod
    def load(cls, path: Path) -> "AthleteProfile":
        return cls(**json.loads(path.read_text()))

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
