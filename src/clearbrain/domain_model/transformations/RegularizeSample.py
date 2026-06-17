# ================================================================
# 0. Section: IMPORTS
# ================================================================
import numpy as np

from dataclasses import dataclass

from .AbstractTransformations import AbstractTransformations
from ..data import ClearVolume, SampleBatch, Atlas, ClearPoints
from .utils import scale_points, get_points_as_volume, resample_to_isotropic



# ================================================================
# 1. Section: Functions
# ================================================================
@dataclass
class RegularizeSample(AbstractTransformations):
    def apply(self, batch: SampleBatch) -> SampleBatch:
        if np.unique(batch.tissue.resolution).size != 1:
            raise ValueError(f"Tissue should be isotropic, right now it is {batch.tissue.resolution}")

        if tuple(batch.tissue.unit) != tuple(batch.cells.unit):
            raise ValueError(f"Tissue unit {batch.tissue.unit} does not match cells unit {batch.cells.unit}")

        atlas = self.convert_atlas(batch.tissue, batch.atlas)
        points = self.convert_points(batch.tissue, batch.cells)

        return batch.copy_with(atlas=atlas, cells=points)


    # ──────────────────────────────────────────────────────
    # 1.1 Subsection: Helper Functions
    # ──────────────────────────────────────────────────────
    def convert_atlas(self, tissue: ClearVolume, atlas: Atlas) -> Atlas:
        data = resample_to_isotropic(atlas.data, atlas.resolution, tissue.resolution[0], True)
        hemispheres = resample_to_isotropic(atlas.hemisphere, atlas.resolution, tissue.resolution[0], True)

        return atlas.copy_with(data=data, hemisphere=hemispheres, resolution=tissue.resolution)

    def convert_points(self, tissue: ClearVolume, points: ClearPoints | ClearVolume) -> ClearVolume:
        if isinstance(points, ClearVolume):
            return points

        scaling_factor = np.round(np.asarray(points.resolution) / np.asarray(tissue.resolution), 5)
        scaled_points = scale_points(points.data, scaling_factor)
        points_as_volume = get_points_as_volume(scaled_points, tissue.shape)

        return ClearVolume(
            data=points_as_volume,
            resolution=tissue.resolution,
            unit=tissue.unit,
            orientation=tissue.orientation,
            tissue_type=tissue.tissue_type,
        )
