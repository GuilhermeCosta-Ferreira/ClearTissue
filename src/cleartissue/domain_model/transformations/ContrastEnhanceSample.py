# ================================================================
# 0. Section: IMPORTS
# ================================================================
import numpy as np
from matplotlib import pyplot as plt

from dataclasses import dataclass

from ..data import SampleBatch, ClearVolume
from .AbstractTransformations import AbstractTransformation



# ================================================================
# 1. Section: Functions
# ================================================================
@dataclass
class ContrastEnhanceSample(AbstractTransformation):
    def apply(self, batch: SampleBatch) -> SampleBatch:
        enhanced_tissue = background_subtraction(batch.tissue)
        enhanced_tissue = clahe(enhanced_tissue)

        return batch.copy_with(tissue=enhanced_tissue)


# ──────────────────────────────────────────────────────
# 1.1 Subsection: Helper Functions
# ──────────────────────────────────────────────────────
def background_subtraction(volume: ClearVolume) -> ClearVolume:
    volume_data = volume.data

    ...

    return volume.copy_with(data=volume_data)

def clahe(volume: ClearVolume) -> ClearVolume:
    volume_data = volume.data

    ...

    return volume.copy_with(data=volume_data)
