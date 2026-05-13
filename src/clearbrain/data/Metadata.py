# ================================================================
# 0. Section: IMPORTS
# ================================================================
from dataclasses import dataclass
from pathlib import Path

from ..tissue.TissueType import TissueType


# ================================================================
# 1. Section: Functions
# ================================================================
@dataclass(frozen=True)
class Metadata:
    mouse: str
    tissue_type: TissueType
    file_path: Path
    description: str = ""

    def __post_init__(self):
        if not isinstance(self.tissue_type, TissueType):
            raise TypeError(
                "Tissue Type needs to be of type TissueType, "
                f"not {type(self.tissue_type)}"
            )

    @property
    def dict(self) -> dict:
        return {
            "mouse": self.mouse,
            "tissue_type": self.tissue_type.as_str,
            "description": self.description,
        }
