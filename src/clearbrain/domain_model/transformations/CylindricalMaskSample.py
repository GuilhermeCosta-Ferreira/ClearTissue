# ================================================================
# 0. Section: IMPORTS
# ================================================================
from dataclasses import dataclass

from ..data import SampleBatch, ClearVolume
from .utils import clear_external_points
from .AbstractTransformations import AbstractTransformation



# ================================================================
# 1. Section: Functions
# ================================================================
@dataclass
class CylindricalMaskSample(AbstractTransformation):
    def apply(self, batch: SampleBatch) -> SampleBatch:
        masked_tissue, angle = clear_external_points(batch.tissue)

        if isinstance(batch.cells, ClearVolume):
            masked_cells, _ = clear_external_points(batch.cells, angle)
        else:
            raise TypeError(f"Expected ClearVolume, got {type(batch.cells)}")

        return batch.copy_with(tissue=masked_tissue, cells=masked_cells)
