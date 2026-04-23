# ================================================================
# 0. Section: IMPORTS
# ================================================================
import numpy as np

from dataclasses import dataclass
from copy import deepcopy

from ..meta import Metadata



# ================================================================
# 1. Section: Functions
# ================================================================
@dataclass
class ClearTissue:
    points: np.ndarray
    metadata: Metadata

    def copy(self, *, deep: bool = True) -> "ClearTissue":
        return ClearTissue(
            points=self.points.copy() if deep else self.points,
            metadata=deepcopy(self.metadata) if deep else self.metadata,
        )
