# ================================================================
# 0. Section: IMPORTS
# ================================================================
import pandas as pd

from numpy.typing import NDArray
from dataclasses import dataclass, replace

from .ClearData import ClearData
from .TissueType import TissueType



# ================================================================
# 1. Section: Functions
# ================================================================
@dataclass
class Atlas(ClearData):
    hemisphere: NDArray
    look_up: pd.DataFrame

    def copy_with(
            self,
            *,
            data: NDArray | None = None,
            resolution: tuple[float, float, float] | None = None,
            unit: tuple[str, str, str] | None = None,
            orientation: str | None = None,
            tissue_type: TissueType | None = None,
            hemisphere: NDArray | None = None,
            look_up: pd.DataFrame | None = None,
        ) -> "Atlas":
            return replace(
                self,
                data=self.data if data is None else data,
                resolution=self.resolution if resolution is None else resolution,
                unit=self.unit if unit is None else unit,
                orientation=self.orientation if orientation is None else orientation,
                tissue_type=self.tissue_type if tissue_type is None else tissue_type,
                hemisphere=self.hemisphere if hemisphere is None else hemisphere,
                look_up=self.look_up if look_up is None else look_up,
            )
