# ================================================================
# 0. Section: IMPORTS
# ================================================================
import numpy as np

from dataclasses import dataclass

from .ClearData import ClearData


# ================================================================
# 1. Section: Functions
# ================================================================
@dataclass
class ClearPoints(ClearData):
    @property
    def estimate_shape(self) -> tuple[int, int, int]:
        nr_axis = self.data.shape[1]

        estimated_shape = []
        for axis in range(nr_axis):
            estimated_shape.append(np.max(self.data[:,axis]))

        return tuple(estimated_shape)
