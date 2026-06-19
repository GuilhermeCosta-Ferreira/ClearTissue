# ================================================================
# 0. Section: IMPORTS
# ================================================================
from dataclasses import dataclass

from ..data import SampleBatch, ClearVolume
from .utils import get_centerline, stretch_tissue
from .AbstractTransformations import AbstractTransformation



# ================================================================
# 1. Section: Functions
# ================================================================
@dataclass
class StretchSample(AbstractTransformation):
    smooth_window_size: int

    def apply(self, batch: SampleBatch) -> SampleBatch:
        centerline = get_centerline(batch.tissue)

        stretched_tissue = stretch_tissue(batch.tissue, centerline, self.smooth_window_size)
        if isinstance(batch.cells, ClearVolume):
            stretched_cell_volume = stretch_tissue(batch.cells, centerline, self.smooth_window_size)
        else:
            raise TypeError(
                f"Expected ClearVolume, got {type(batch.cells)} for the cells,"
                " make sure you run RegularizeSample first"
            )

        return batch.copy_with(tissue=stretched_tissue, cells=stretched_cell_volume)
