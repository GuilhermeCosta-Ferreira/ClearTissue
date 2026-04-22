# ================================================================
# 0. Section: IMPORTS
# ================================================================
from dataclasses import dataclass
from pathlib import Path

from .DataType import DataType
from .tissue_loader import load_points



# ================================================================
# 1. Section: Functions
# ================================================================
@dataclass(frozen=True)
class LoadTissue:
    path: Path

    def load_tissue(self, type: DataType = DataType.POINTS):
        if type == DataType.POINTS:
            return self._load_points()
        else:
            return self._load_volume()


    def _load_points(self):
        return load_points(self.path)

    def _load_volume(self):
        pass
