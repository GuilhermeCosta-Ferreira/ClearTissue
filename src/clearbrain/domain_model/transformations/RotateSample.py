# ================================================================
# 0. Section: IMPORTS
# ================================================================
from dataclasses import dataclass

from .utils import rotate_spinal_cord
from ..data import SampleBatch, ClearVolume
from .AbstractTransformations import AbstractTransformation



# ================================================================
# 1. Section: Functions
# ================================================================
@dataclass
class RotateSample(AbstractTransformation):
    def apply(self, batch: SampleBatch) -> SampleBatch:
        rotated_tissue, angle = rotate_spinal_cord(batch.tissue, batch.atlas)

        if isinstance(batch.cells, ClearVolume):
            rotated_cells, _ = rotate_spinal_cord(batch.cells, angle=angle)
        else:
            raise TypeError(f"Expected ClearVolume, got {type(batch.cells)}")

        return batch.copy_with(tissue=rotated_tissue, cells=rotated_cells)
