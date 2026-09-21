# ================================================================
# 0. Section: IMPORTS
# ================================================================
from dataclasses import dataclass

from ..data import SampleBatch, ClearVolume
from .utils import crop_excess, apply_crop_excess
from .AbstractTransformations import AbstractTransformation



# ================================================================
# 1. Section: Functions
# ================================================================
@dataclass
class EmptySpaceTrimSample(AbstractTransformation):
    def apply(self, batch: SampleBatch) -> SampleBatch:
        cropped_tissue, crop_params = crop_excess(batch.tissue)

        if isinstance(batch.cells, ClearVolume):
            cropped_cells = apply_crop_excess(batch.cells, crop_params)
        else:
            raise TypeError(f"Expected ClearVolume, got {type(batch.cells)}")

        return batch.copy_with(tissue=cropped_tissue, cells=cropped_cells)
