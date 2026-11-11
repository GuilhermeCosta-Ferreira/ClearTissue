# ================================================================
# 0. Section: IMPORTS
# ================================================================
import numpy as np

from numpy.typing import NDArray
from dataclasses import dataclass

from .AbstractProfileAlignment import AbstractProfileAlignment
from ..analysis.cord_profile import Direction, elongation_profile, find_atlas_shift


# ================================================================
# 1. Section: Functions
# ================================================================
@dataclass
class CorrelationAlignment(AbstractProfileAlignment):
    """Align by cross-correlating the cord's width profile with the atlas's.

    The atlas is cropped / zero-padded along axis 0 so tissue slice i lines up
    with atlas slice i (see AbstractProfileAlignment).

    Attributes:
        direction: In-plane axis the width profile is measured along.
    """

    direction: Direction = "horizontal"

    def find_shift(
        self, tissue: NDArray, atlas: NDArray, excluded: NDArray | None
    ) -> int:
        tissue_profile = elongation_profile(tissue, self.direction)
        if excluded is not None:
            tissue_profile[excluded] = np.nan  # NaN slices drop out of the correlation
        return find_atlas_shift(
            tissue_profile, elongation_profile(atlas, self.direction)
        )
