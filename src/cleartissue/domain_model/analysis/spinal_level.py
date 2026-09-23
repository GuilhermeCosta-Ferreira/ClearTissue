# ================================================================
# 0. Section: IMPORTS
# ================================================================
from collections import defaultdict

import numpy as np
import pandas as pd

from numpy.typing import NDArray

from ..data import Atlas, ClearVolume, ClearPoints



# ================================================================
# 1. Section: Constants
# ================================================================
# Orientation letters (BrainGlobe) whose axis runs rostro-caudal for the cord.
_ROSTRO_CAUDAL_LETTERS: frozenset[str] = frozenset({"a", "p"})
BACKGROUND_LABEL: int = 0



# ================================================================
# 2. Section: Functions
# ================================================================
def count_spinal_level_overlap(
    cells: ClearVolume | ClearPoints,
    atlas: Atlas,
    segments: list[dict],
) -> pd.DataFrame:
    axis = _section_axis(atlas)
    annotation = np.moveaxis(atlas.data, axis, 0)
    bounds = _level_bounds(segments, annotation.shape[0])

    cell_axis_pos, cell_mask = _cells_on_axis(cells, annotation, atlas, axis)

    rows: list[dict] = []
    for order, (name, start, end) in enumerate(bounds):
        slab = annotation[start:end]
        volume = int(np.count_nonzero(slab))

        in_slab = (cell_axis_pos >= start) & (cell_axis_pos < end)
        cell_sum = int(np.count_nonzero(cell_mask & in_slab))

        rows.append(
            {
                "order": order,
                "segment": name,
                "present": volume > 0,
                "cell_sum": cell_sum,
                "volume": volume,
                "density_pct": 100.0 * cell_sum / volume if volume else 0.0,
            }
        )

    return pd.DataFrame(
        rows,
        columns=["order", "segment", "present", "cell_sum", "volume", "density_pct"],
    )

def region_spinal_levels(
    atlas: Atlas,
    segments: list[dict],
    structures: dict[int, dict],
) -> dict[int, list[str]]:
    axis = _section_axis(atlas)
    annotation = np.moveaxis(atlas.data, axis, 0)
    bounds = _level_bounds(segments, annotation.shape[0])
    order = {name: i for i, (name, _, _) in enumerate(bounds)}

    aggregated: dict[int, set[str]] = defaultdict(set)
    for name, start, end in bounds:
        for label in np.unique(annotation[start:end]):
            label = int(label)
            if label == BACKGROUND_LABEL:
                continue
            path = structures.get(label, {}).get("structure_id_path", [label])
            for region_id in path:
                aggregated[int(region_id)].add(name)

    return {rid: sorted(names, key=order.get) for rid, names in aggregated.items()}


# ──────────────────────────────────────────────────────
# 1.1 Subsection: Helper Functions
# ──────────────────────────────────────────────────────
def _section_axis(atlas: Atlas) -> int:
    for axis, letter in enumerate(atlas.orientation):
        if letter in _ROSTRO_CAUDAL_LETTERS:
            return axis
    return int(np.argmax(atlas.data.shape))

def _level_bounds(segments: list[dict], axis_len: int) -> list[tuple[str, int, int]]:
    total_sections = max(int(s["End"]) for s in segments)

    bounds: list[tuple[str, int, int]] = []
    for segment in segments:
        start = int(round((int(segment["Start"]) - 1) / total_sections * axis_len))
        end = int(round(int(segment["End"]) / total_sections * axis_len))
        bounds.append((str(segment["Segment"]), start, max(end, start + 1)))
    return bounds

def _cells_on_axis(
    cells: ClearVolume | ClearPoints,
    annotation: NDArray,
    atlas: Atlas,
    axis: int,
) -> tuple[NDArray, NDArray]:
    if isinstance(cells, ClearPoints):
        coords = np.rint(cells.data).astype(np.int64)
        in_bounds = np.all((coords >= 0) & (coords < atlas.data.shape), axis=1)
        coords = coords[in_bounds]

        axis_pos = coords[:, axis]
        labels = atlas.data[coords[:, 0], coords[:, 1], coords[:, 2]]
        return axis_pos, labels != BACKGROUND_LABEL

    if cells.data.shape != atlas.data.shape:
        raise ValueError(
            f"Cells volume shape {cells.data.shape} does not match "
            f"atlas annotation shape {atlas.data.shape}"
        )
    presence = (np.moveaxis(cells.data, axis, 0) > 0) & (annotation != BACKGROUND_LABEL)
    axis_pos = np.indices(annotation.shape)[0]
    return axis_pos.ravel(), presence.ravel()
