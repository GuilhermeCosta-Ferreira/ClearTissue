# ================================================================
# 0. Section: IMPORTS
# ================================================================
from dataclasses import dataclass

from .ClearData import ClearData


# ================================================================
# 1. Section: Functions
# ================================================================
@dataclass
class ClearVolume(ClearData):
    @property
    def shape(self) -> tuple[int, int, int]:
        return self.data.shape
