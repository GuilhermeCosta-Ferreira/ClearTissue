# ================================================================
# 0. Section: IMPORTS
# ================================================================
from typing import Sequence

import numpy as np
from numpy.typing import NDArray

_AXIS_GROUP = {
    "a": "ap",
    "p": "ap",
    "s": "si",
    "i": "si",
    "r": "rl",
    "l": "rl",
}



# ================================================================
# 1. Section: Functions
# ================================================================
def reorient_array(
    data: NDArray,
    source_orientation: str,
    target_orientation: str,
) -> NDArray:
    permutation, flips = get_orientation_transform(
        source_orientation,
        target_orientation,
    )

    reoriented = np.transpose(data, axes=permutation)

    for axis, should_flip in enumerate(flips):
        if should_flip:
            reoriented = np.flip(reoriented, axis=axis)

    return reoriented

def reorient_tuple(
    values: Sequence,
    source_orientation: str,
    target_orientation: str,
) -> tuple:
    permutation, _ = get_orientation_transform(
        source_orientation,
        target_orientation,
    )

    return tuple(values[i] for i in permutation)


# ──────────────────────────────────────────────────────
# 1.1 Subsection: Helper Functions
# ──────────────────────────────────────────────────────
def get_orientation_transform(
    source_orientation: str,
    target_orientation: str,
) -> tuple[tuple[int, int, int], tuple[bool, bool, bool]]:
    source_groups = [_AXIS_GROUP[axis] for axis in source_orientation]

    permutation: list[int] = []
    flips: list[bool] = []

    for target_axis in target_orientation:
        target_group = _AXIS_GROUP[target_axis]

        source_axis_index = source_groups.index(target_group)

        permutation.append(source_axis_index)
        flips.append(source_orientation[source_axis_index] != target_axis)

    return tuple(permutation), tuple(flips)  # type: ignore
