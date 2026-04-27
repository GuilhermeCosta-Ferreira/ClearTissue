# ================================================================
# 0. Section: IMPORTS
# ================================================================
from dataclasses import dataclass
from pathlib import Path

from ..tissue import TissueType



# ================================================================
# 1. Section: Functions
# ================================================================
@dataclass(frozen=True)
class Metadata:
    mouse: str
    tissue_type: TissueType
    file_path: Path
    description: str = ""

    @property
    def dict(self) -> dict:
        return {
            "mouse": self.mouse,
            "tissue_type": self.tissue_type.str,
            "description": self.description
        }
