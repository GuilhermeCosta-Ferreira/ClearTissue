# ================================================================
# 0. Section: IMPORTS
# ================================================================
from typing import Self
from numpy.typing import NDArray
from dataclasses import dataclass, replace

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

    def copy_with(
            self,
            *,
            data: NDArray | None = None,
            resolution: tuple[float, float, float] | None = None,
            unit: tuple[str, str, str] | None = None,
            orientation: str | None = None,
            tissue_type: TissueType | None = None,
        ) -> Self:
            return replace(
                self,
                data=self.data if data is None else data,
                resolution=self.resolution if resolution is None else resolution,
                unit=self.unit if unit is None else unit,
                orientation=self.orientation if orientation is None else orientation,
                tissue_type=self.tissue_type if tissue_type is None else tissue_type,
            )
