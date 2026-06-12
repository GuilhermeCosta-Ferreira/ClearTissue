# ================================================================
# 0. Section: IMPORTS
# ================================================================
from numpy.typing import NDArray
from dataclasses import dataclass

from .TissueType import TissueType



# ================================================================
# 1. Section: Functions
# ================================================================
@dataclass
class ClearData:
    data: NDArray
    resolution: tuple[float, float, float]
    unit: tuple[str, str, str]
    orientation: str
    tissue_type: TissueType
