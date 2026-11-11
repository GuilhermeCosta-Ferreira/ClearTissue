# ================================================================
# 0. Section: IMPORTS
# ================================================================
import numpy as np

from dataclasses import dataclass

from ..data import SampleBatch
from .AbstractTransformations import AbstractTransformation
from ..analysis.cord_profile import keep_main_component


# ================================================================
# 1. Section: Functions
# ================================================================
@dataclass
class MainComponentCleanSample(AbstractTransformation):
    """Keep only the cord's main blob per slice, cutting debris off by erosion.

    Unlike CleanDebrisTransformation (3D, no erosion), each slice is eroded so
    thin bridges to debris break, the largest 2D component is kept, and with
    restore it is dilated back inside the original mask. Original intensities
    are kept inside the mask. Meant as preprocessing before warping.

    Attributes:
        erosion_iterations: Erosion steps before picking the largest component.
        restore: Dilate the kept component back by the same amount.
    """

    erosion_iterations: int = 2
    restore: bool = True

    def apply(self, batch: SampleBatch) -> SampleBatch:
        tissue = batch.tissue.data
        kept = keep_main_component(tissue, self.erosion_iterations, self.restore)
        return batch.copy_with(
            tissue=batch.tissue.copy_with(data=np.where(kept, tissue, 0))
        )
