# ================================================================
# 0. Section: IMPORTS
# ================================================================
from dataclasses import dataclass

from ...registration import Registrator
from ..data import SampleBatch, ClearVolume
from .AbstractTransformations import AbstractTransformation
from .utils import untwist_spinal_coord, apply_know_untwisting



# ================================================================
# 1. Section: Functions
# ================================================================
@dataclass
class UntwistSample(AbstractTransformation):
    tissue_registrator: Registrator
    cell_registrator: Registrator
    window_size: int
    gap_size: int

    def apply(self, batch: SampleBatch) -> SampleBatch:
        untwisted_tissue, twisting_data = untwist_spinal_coord(
            tissue = batch.tissue,
            registrator = self.tissue_registrator,
            window_size = self.window_size,
            gap = self.gap_size
        )

        if isinstance(batch.cells, ClearVolume):
            untwisted_cell = apply_know_untwisting(
                batch.cells,
                self.cell_registrator,
                twisting_data
            )
        else:
            raise TypeError(
                f"Expected ClearVolume, got {type(batch.cells)} instead"
            )

        return batch.copy_with(tissue=untwisted_tissue, cells=untwisted_cell)
