# ================================================================
# 0. Section: IMPORTS
# ================================================================
from __future__ import annotations

import numpy as np

from dataclasses import dataclass
from copy import deepcopy

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..data import Metadata


# ================================================================
# 1. Section: Functions
# ================================================================
@dataclass
class ClearVolume:
    volume: np.ndarray
    metadata: Metadata
    sample_factor: int

    def copy(self, *, deep: bool = True) -> "ClearVolume":
        return ClearVolume(
            volume=self.volume.copy() if deep else self.volume,
            metadata=deepcopy(self.metadata) if deep else self.metadata,
            sample_factor=self.sample_factor,
        )
