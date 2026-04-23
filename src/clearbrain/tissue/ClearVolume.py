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
class ClearVolume:
    volume: np.ndarray
    metadata: Metadata

    def copy(self, *, deep: bool = True) -> "ClearVolume":
        return ClearVolume(
            volume=self.volume.copy() if deep else self.volume,
            metadata=deepcopy(self.metadata) if deep else self.metadata,
        )
