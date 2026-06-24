# ================================================================
# 0. Section: IMPORTS
# ================================================================
import numpy as np

from dataclasses import dataclass

from ..data import SampleBatch
from .AbstractTransformations import AbstractTransformation



# ================================================================
# 1. Section: Functions
# ================================================================
@dataclass
class PruneAtlas(AbstractTransformation):
    def apply(self, batch: SampleBatch) -> SampleBatch:
        atlas_data = batch.atlas.data
        tissue = batch.tissue.data

        pruned_atlas_data = np.where(tissue > 0, atlas_data, 0)
        pruned_atlas = batch.atlas.copy_with(data=pruned_atlas_data)

        return batch.copy_with(atlas=pruned_atlas)
