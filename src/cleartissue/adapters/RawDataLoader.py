# ================================================================
# 0. Section: IMPORTS
# ================================================================
import h5py
import json

import numpy as np

from numpy.typing import NDArray
from dataclasses import dataclass

from .Source import Source
from .utils import get_json_cell_data, get_raw_tissue_path
from ..domain_model.data import ClearPoints, Atlas, ClearVolume, TissueType



# ================================================================
# 1. Section: Functions
# ================================================================
@dataclass
class RawDataLoader:
    source: Source

    def load_raw_cells(
        self,
        resolution: tuple[float, float, float],
        unit: tuple[str, str, str],
        orientation: str,
    ) -> ClearPoints:
        data = get_json_cell_data(self.source.raw_path, self.source.cells_base_name)

        return ClearPoints(
            data = np.asarray(data),
            resolution = resolution,
            unit = unit,
            orientation = orientation,
            tissue_type = self.source.tissue_type
        )

    def load_raw_atlas(self, unit: tuple[str, str, str]) -> Atlas:
        return Atlas.from_name(
            self.source.atlas_name,
            unit = unit,
            tissue_type = self.source.tissue_type
        )

    def load_raw_tissue(self) -> ClearVolume:
        h5_path = get_raw_tissue_path(self.source.raw_path, self.source.tissue_base_name)

        with h5py.File(h5_path, "r") as f:
            data: NDArray = np.asarray(f["data"])

            resolution = tuple(json.loads(f.attrs["resolution"]))  # type: ignore
            unit = tuple(json.loads(f.attrs["unit"]))  # type: ignore
            orientation = str(f.attrs["orientation"])
            tissue_type = TissueType[str(f.attrs["tissue_type"])]

        return ClearVolume(
            data=data,
            resolution=resolution,
            unit=unit,
            orientation=orientation,
            tissue_type=tissue_type,
        )
