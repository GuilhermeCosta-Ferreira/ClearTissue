"""Per-section position of Clarke's column in the Allen spinal-cord atlas.

For every spinal section (C1, ..., L1, ...) of the atlas that contains
Clarke's column (dorsal nucleus), this script averages the column's centroid
over all atlas slices of that section and expresses it relative to the cord:

    x_um: lateral distance from the cord midline (0); the left column is
        mirrored onto the right, so this is positive (right side).
    y_um: dorso-ventral offset from the dorsal cord surface (0), ventral < 0.

The midline is the left-right centroid of the cord mask in that section and
the top is the dorsal-most cord voxel in that section, so both references
follow the cord's shape along its length; every slice weighs equally in
the section average. Edit the inputs below in place.

Output is one ``.csv`` with one row per spinal section.
"""

# ================================================================
# 0. Section: IMPORTS
# ================================================================
from typing import Final
from pathlib import Path

import numpy as np
import pandas as pd

from numpy.typing import NDArray
from brainglobe_atlasapi import BrainGlobeAtlas


# ================================================================
# 1. Section: INPUTS
# ================================================================
# Final keeps the Literal type BrainGlobeAtlas expects for its name.
ATLAS_NAME: Final = "allen_cord_20um"
CLARKE_ACRONYM: str = "D"
OUTPUT_PATH: Path = Path("out/clarke_position.csv")

_BACKGROUND_LABEL: int = 0
_EXPECTED_ORIENTATION: str = "asr"


# ================================================================
# 2. Section: FUNCTIONS
# ================================================================
def clarke_positions(atlas: BrainGlobeAtlas) -> pd.DataFrame:
    """Return Clarke's column position averaged per spinal section.

    Returns:
        One row per spinal section, in atlas order, with ``segment``,
        ``x_um``, ``y_um`` and ``n_slices`` (atlas slices averaged).
    """
    per_slice = _slice_positions(atlas)
    return (
        per_slice.groupby("segment", sort=False)
        .agg(
            x_um=("x_um", "mean"),
            y_um=("y_um", "mean"),
            n_slices=("slice", "nunique"),
        )
        .round(2)
        .reset_index()
    )


# ──────────────────────────────────────────────────────
# 2.1 Subsection: Helper Functions
# ──────────────────────────────────────────────────────
def _slice_positions(atlas: BrainGlobeAtlas) -> pd.DataFrame:
    if atlas.orientation != _EXPECTED_ORIENTATION:
        raise ValueError(
            f"Expected orientation '{_EXPECTED_ORIENTATION}', got '{atlas.orientation}'."
        )

    annotation = np.asarray(atlas.annotation)
    clarke_id = atlas.structures[CLARKE_ACRONYM]["id"]
    _, dv_um, lr_um = atlas.resolution
    segment_of = _section_to_segment(atlas.metadata.get("atlas_segments", []))

    rows: list[dict] = []
    present = np.flatnonzero((annotation == clarke_id).any(axis=(1, 2)))
    for section in map(int, present):
        plane = annotation[section]
        cord = plane != _BACKGROUND_LABEL
        clarke = plane == clarke_id

        midline = _centroid(cord)[1]
        top = np.flatnonzero(cord.any(axis=1))[0]

        for half in _split_sides(clarke, midline):
            if not half.any():
                continue
            dv, lr = _centroid(half)
            rows.append(
                {
                    "segment": segment_of.get(section + 1, ""),
                    "slice": section + 1,
                    # Sides mirror each other, so average lateral distance.
                    "x_um": float(abs(midline - lr) * lr_um),
                    "y_um": float(-(dv - top) * dv_um),
                }
            )

    return pd.DataFrame(rows)


def _section_to_segment(segments: list[dict]) -> dict[int, str]:
    return {
        section: seg["Segment"]
        for seg in segments
        for section in range(seg["Start"], seg["End"] + 1)
    }


def _centroid(mask: NDArray) -> NDArray:
    return np.argwhere(mask).mean(axis=0)


def _split_sides(mask: NDArray, midline: float) -> tuple[NDArray, NDArray]:
    lr_index = np.arange(mask.shape[1])[np.newaxis, :]
    return mask & (lr_index < midline), mask & (lr_index >= midline)


# ================================================================
# 3. Section: MAIN
# ================================================================
def main() -> None:
    atlas = BrainGlobeAtlas(ATLAS_NAME, check_latest=False)
    positions = clarke_positions(atlas)
    positions.to_csv(OUTPUT_PATH, index=False)
    print(f"Wrote {len(positions)} rows to {OUTPUT_PATH.resolve()}")


if __name__ == "__main__":
    main()
