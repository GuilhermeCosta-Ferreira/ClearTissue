# ================================================================
# 0. Section: IMPORTS
# ================================================================
from dataclasses import dataclass

from ..data import SampleBatch, ClearVolume
from .AbstractTransformations import AbstractTransformation



# ================================================================
# 1. Section: Functions
# ================================================================
@dataclass
class StartEndTransformtaion(AbstractTransformation):
    start_slice: int
    end_slice: int

    def apply(self, batch: SampleBatch) -> SampleBatch:
        cropped_tissue = crop_ends(batch.tissue, self.start_slice, self.end_slice)

        if isinstance(batch.cells, ClearVolume):
            cropped_cell_volume = crop_ends(batch.cells, self.start_slice, self.end_slice)
        else:
            raise TypeError(
                f"Expected ClearVolume, got {type(batch.cells)} for the cells,"
                " make sure you run RegularizeSample first"
            )

        return batch.copy_with(tissue=cropped_tissue, cells=cropped_cell_volume)


def crop_ends(volume: ClearVolume, start_slice: int, end_slice: int) -> ClearVolume:
    cropped_data = volume.data[start_slice:end_slice]
    return volume.copy_with(data=cropped_data)
