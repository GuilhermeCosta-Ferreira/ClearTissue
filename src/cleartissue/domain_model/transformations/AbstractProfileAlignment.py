# ================================================================
# 0. Section: IMPORTS
# ================================================================
from abc import abstractmethod
from numpy.typing import NDArray
from dataclasses import dataclass

from ..data import SampleBatch
from .utils.alignment import apply_atlas_shift
from .AbstractTransformations import AbstractTransformation


# ================================================================
# 1. Section: Functions
# ================================================================
@dataclass
class AbstractProfileAlignment(AbstractTransformation):
    """Shift the atlas along axis 0 so its rostro-caudal profile meets the cord.

    Subclasses only decide how a shift is scored (find_shift); applying it is
    shared: the atlas is cropped / zero-padded to the tissue's length so tissue
    slice i lines up with atlas slice i (see apply_atlas_shift).
    """

    def apply(self, batch: SampleBatch) -> SampleBatch:
        shift = self.find_shift(batch.tissue.data, batch.atlas.data, None)
        print(f"{type(self).__name__}: atlas shift {shift}")
        return apply_atlas_shift(batch, shift)

    @abstractmethod
    def find_shift(
        self, tissue: NDArray, atlas: NDArray, excluded: NDArray | None
    ) -> int:
        """Return s such that tissue slice i sits on atlas slice i + s.

        Implementations must ignore tissue slices where excluded is True
        (None = use every slice), so BumpExcludedAlignment can compose them.
        """
        raise NotImplementedError
