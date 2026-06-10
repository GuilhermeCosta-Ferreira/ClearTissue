# ================================================================
# 0. Section: IMPORTS
# ================================================================
import numpy as np

from pathlib import Path
from numpy.typing import NDArray
from dataclasses import dataclass

from ..tissue.TissueType import TissueType



# ================================================================
# 1. Section: Functions
# ================================================================
@dataclass(frozen=False)
class Metadata:
    mouse: str
    tissue_type: TissueType
    file_path: Path
    scale_factor: tuple[float, float, float] | NDArray = (1.0, 1.0, 1.0)
    description: str = ""

    def __post_init__(self):
        if not isinstance(self.tissue_type, TissueType):
            raise TypeError(
                "Tissue Type needs to be of type TissueType, "
                f"not {type(self.tissue_type)}"
            )

        if isinstance(self.scale_factor, np.ndarray):
            self.scale_factor = tuple(self.scale_factor)

    @property
    def dict(self) -> dict:
        return {
            "mouse": self.mouse,
            "tissue_type": self.tissue_type.as_str,
            "scale_factor": self.scale_factor,
            "description": self.description,
        }
