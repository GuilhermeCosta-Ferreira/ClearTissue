# ================================================================
# 0. Section: IMPORTS
# ================================================================
from dataclasses import dataclass
import json
import os
from pathlib import Path

from ..tissue import TissueType
from .load_metadata import load_metadata




# ================================================================
# 1. Section: Functions
# ================================================================
@dataclass(frozen=True)
class Metadata:
    mouse: str
    tissue_type: TissueType
    description: str = ""

    @classmethod
    def from_path(cls, path: Path) -> 'Metadata':
        meta = load_metadata(path)

        return Metadata(
            mouse = meta["mouse"],
            tissue_type = meta["tissue_type"],
            description = meta["description"]
        )

    @classmethod
    def from_filename(cls, path: Path):
        tissue_type = path.stem.rsplit("_", 2)[-2]

        return Metadata(
            mouse = str(path.parent.name),
            tissue_type=TissueType.from_str(tissue_type),
        )

    @property
    def dict(self) -> dict:
        return {
            "mouse": self.mouse,
            "tissue_type": self.tissue_type.str,
            "description": self.description
        }


    def create_metadata(self, path: Path) -> None:
        if os.path.exists(path):
            raise FileExistsError(f"There is already a metadata file here ({str(path)})")

        with path.open("w", encoding="utf-8") as f:
            json.dump(self.dict, f, indent=2)
