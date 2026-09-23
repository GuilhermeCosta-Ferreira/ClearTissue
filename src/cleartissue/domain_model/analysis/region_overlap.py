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
BACKGROUND_LABEL: int = 0



# ================================================================
# 2. Section: Functions
# ================================================================
def count_region_overlap(
    cells: ClearVolume | ClearPoints,
    atlas: Atlas,
    structures: dict[int, dict],
) -> pd.DataFrame:
    cell_sums = _aggregate_up(_direct_cell_counts(cells, atlas.data), structures)
    volumes = _aggregate_up(_direct_voxel_counts(atlas.data), structures)
    children = _direct_children(structures)

    rows: list[dict] = []
    for label, structure in structures.items():
        path = structure["structure_id_path"]
        parent_id = int(path[-2]) if len(path) > 1 else None
        parent_acronym = (
            structures[parent_id]["acronym"]
            if parent_id is not None and parent_id in structures
            else ""
        )

        volume = volumes.get(label, 0)
        cell_sum = cell_sums.get(label, 0)
        child_acronyms = [
            structures[c]["acronym"] for c in children.get(label, []) if c in structures
        ]

        rows.append(
            {
                "id": label,
                "name": structure["name"],
                "acronym": structure["acronym"],
                "parent": parent_acronym,
                "present": volume > 0,
                "depth": len(path) - 1,
                "cell_sum": cell_sum,
                "volume": volume,
                "density_pct": 100.0 * cell_sum / volume if volume else 0.0,
                "children": ";".join(child_acronyms),
            }
        )

    frame = pd.DataFrame(
        rows,
        columns=[
            "id",
            "name",
            "acronym",
            "parent",
            "present",
            "depth",
            "cell_sum",
            "volume",
            "density_pct",
            "children",
        ],
    )
    return frame.sort_values("cell_sum", ascending=False).reset_index(drop=True)


# ──────────────────────────────────────────────────────
# 1.1 Subsection: Helper Functions
# ──────────────────────────────────────────────────────
def _direct_cell_counts(
    cells: ClearVolume | ClearPoints,
    annotation: NDArray,
) -> dict[int, int]:
    annotation = annotation.astype(np.int64, copy=False)
    n_labels = int(annotation.max()) + 1

    if isinstance(cells, ClearPoints):
        coords = np.rint(cells.data).astype(np.int64)

        # Keep only points that fall inside the annotation volume.
        in_bounds = np.all((coords >= 0) & (coords < annotation.shape), axis=1)
        coords = coords[in_bounds]

        labels = annotation[coords[:, 0], coords[:, 1], coords[:, 2]]
        counts = np.bincount(labels.ravel(), minlength=n_labels)
    else:
        if cells.data.shape != annotation.shape:
            raise ValueError(
                f"Cells volume shape {cells.data.shape} does not match "
                f"atlas annotation shape {annotation.shape}"
            )
        presence = (cells.data > 0).astype(np.int64)
        counts = np.bincount(
            annotation.ravel(), weights=presence.ravel(), minlength=n_labels
        ).astype(np.int64)

    counts[BACKGROUND_LABEL] = 0
    return {label: int(c) for label, c in enumerate(counts) if c}


def _direct_voxel_counts(annotation: NDArray) -> dict[int, int]:
    counts = np.bincount(annotation.astype(np.int64, copy=False).ravel())
    counts[BACKGROUND_LABEL] = 0
    return {label: int(c) for label, c in enumerate(counts) if c}


def _aggregate_up(
    direct: dict[int, int],
    structures: dict[int, dict],
) -> dict[int, int]:
    aggregated: dict[int, int] = defaultdict(int)
    for label, value in direct.items():
        structure = structures.get(label)
        if structure is None:
            continue
        for ancestor in structure["structure_id_path"]:
            aggregated[int(ancestor)] += value
    return aggregated


def _direct_children(structures: dict[int, dict]) -> dict[int, list[int]]:
    children: dict[int, list[int]] = defaultdict(list)
    for label, structure in structures.items():
        path = structure["structure_id_path"]
        if len(path) > 1:
            children[int(path[-2])].append(int(label))
    return children
