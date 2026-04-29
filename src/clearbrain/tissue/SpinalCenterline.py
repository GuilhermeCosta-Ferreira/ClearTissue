# ================================================================
# 0. Section: IMPORTS
# ================================================================
import numpy as np

from dataclasses import dataclass

from ..data import Metadata



# ================================================================
# 1. Section: Functions
# ================================================================
@dataclass(frozen=True)
class SpinalCenterline:
    volume: np.ndarray
    points: np.ndarray
    metadata: Metadata
