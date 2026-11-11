# ================================================================
# 0. Section: IMPORTS
# ================================================================
import numpy as np

from numpy.typing import NDArray

from ...data import SampleBatch


# ================================================================
# 1. Section: Functions
# ================================================================
def apply_atlas_shift(batch: SampleBatch, shift: int) -> SampleBatch:
    """Re-index the atlas along axis 0 so tissue slice i sits on atlas slice i.

    shift follows find_atlas_shift: tissue slice i matches atlas slice
    i + shift. The atlas (labels, hemisphere and template) is cropped and
    zero-padded to the tissue's length; the tissue is left untouched.
    """
    n_slices = batch.tissue.data.shape[0]
    atlas = batch.atlas.copy_with(
        data=_shift_axis0(batch.atlas.data, shift, n_slices),
        hemisphere=_shift_axis0(batch.atlas.hemisphere, shift, n_slices),
        tissue_template=_shift_axis0(batch.atlas.tissue_template, shift, n_slices),
    )
    return batch.copy_with(atlas=atlas)


# ──────────────────────────────────────────────────────
# 1.1 Subsection: Helper Functions
# ──────────────────────────────────────────────────────
def _shift_axis0(volume: NDArray, shift: int, n_slices: int) -> NDArray:
    out = np.zeros((n_slices,) + volume.shape[1:], dtype=volume.dtype)
    source = np.arange(n_slices) + shift
    inside = (source >= 0) & (source < volume.shape[0])
    out[inside] = volume[source[inside]]
    return out
