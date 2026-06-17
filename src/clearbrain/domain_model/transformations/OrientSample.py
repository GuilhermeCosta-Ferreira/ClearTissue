# ================================================================
# 0. Section: IMPORTS
# ================================================================
from dataclasses import dataclass

from ..data import SampleBatch, ClearVolume
from .AbstractTransformations import AbstractTransformation
from .utils import reorient_array, reorient_tuple



# ================================================================
# 1. Section: Functions
# ================================================================
@dataclass
class OrientSample(AbstractTransformation):
    def apply(self, batch: SampleBatch) -> SampleBatch:
        target_orientation = batch.atlas.orientation

        tissue = self._orient_volume(batch.tissue, target_orientation)
        if isinstance(batch.cells, ClearVolume):
            cells = self._orient_volume(batch.cells, target_orientation)
        else:
            print(f"Cells need to be a ClearVolume for re-orientation, got {type(batch.cells)} instead")
            cells = batch.cells

        return batch.copy_with(tissue=tissue, cells=cells)


    def _orient_volume(self, clear_volume: ClearVolume, target_orientation: str) -> ClearVolume:
        source_orientation = clear_volume.orientation

        if source_orientation == target_orientation:
            return clear_volume

        reoriented_tissue = clear_volume.copy_with(
            data=reorient_array(
                clear_volume.data,
                source_orientation,
                target_orientation,
            ),
            resolution=reorient_tuple(
                clear_volume.resolution,
                source_orientation,
                target_orientation,
            ),
            unit=reorient_tuple(
                clear_volume.unit,
                source_orientation,
                target_orientation,
            ),
            orientation=target_orientation,
        )

        return reoriented_tissue
