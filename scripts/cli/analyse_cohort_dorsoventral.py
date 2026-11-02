"""Cohort Excel of cell overlap split dorsal/ventral and by spinal level.

For every mouse in the cohort this script finds the spinal-cord atlas centre
of gravity along the dorso-ventral axis, cuts the cord into a dorsal
(superior) and a ventral (inferior) half at that plane, and then counts cell
overlap per spinal level within each half — giving ``dorsal L1``, ``ventral
L1``, ... for each animal.

All of the dorsal/ventral geometry lives in this script; it reuses the
packaged loaders and ``count_spinal_level_overlap`` (which already accepts a
boolean mask) rather than adding anything to the library. Edit the inputs
below in place — this is a cohort script, not a CLI.

Output is one ``.xlsx`` with two sheets:
    level_dorsoventral: one row per (animal, spinal level), dorsal and ventral
        cell_sum / volume / density_pct side by side.
    summary_dorsoventral: one row per animal, dorsal vs ventral totals plus the
        centre-of-gravity cut position.
"""

# ================================================================
# 0. Section: IMPORTS
# ================================================================
from pathlib import Path
from dataclasses import dataclass

import numpy as np
import pandas as pd

from numpy.typing import NDArray

from cleartissue.domain_model.data import Atlas, TissueType
from cleartissue.adapters.RawDataLoader import RawDataLoader
from cleartissue.service.ClearTissueProject import ClearTissueProject
from cleartissue.domain_model.analysis import count_spinal_level_overlap


# ================================================================
# 1. Section: Types
# ================================================================
@dataclass(frozen=True)
class AnimalRun:
    mouse: str
    pipeline_id: int
    step_id: int


# ================================================================
# 2. Section: Inputs
# ================================================================
# Edit these in place — this is a cohort script, not a CLI.
TISSUE_TYPE: TissueType = TissueType.SPINAL_CORD

# BrainGlobe orientation letters whose axis runs dorsal(superior)-ventral.
_DORSOVENTRAL_LETTERS: frozenset[str] = frozenset({"s", "i"})
_SUPERIOR_LETTER: str = "s"
_BACKGROUND_LABEL: int = 0
_VALUE_COLS: tuple[str, ...] = ("cell_sum", "volume", "density_pct")

ANIMALS: list[AnimalRun] = [
    AnimalRun(mouse="31E", pipeline_id=3, step_id=8),
    AnimalRun(mouse="32E", pipeline_id=2, step_id=8),
    AnimalRun(mouse="33E", pipeline_id=2, step_id=8),
    AnimalRun(mouse="79E", pipeline_id=2, step_id=8),
    AnimalRun(mouse="116Rest", pipeline_id=2, step_id=8),
    AnimalRun(mouse="117Rest", pipeline_id=2, step_id=8),
    AnimalRun(mouse="120Rest", pipeline_id=2, step_id=8),
    AnimalRun(mouse="121Rest", pipeline_id=2, step_id=8),
    AnimalRun(mouse="122Rest", pipeline_id=2, step_id=8),
    AnimalRun(mouse="189R", pipeline_id=2, step_id=8),
    AnimalRun(mouse="191R", pipeline_id=2, step_id=8),
    AnimalRun(mouse="193R", pipeline_id=2, step_id=8),
    AnimalRun(mouse="194B", pipeline_id=2, step_id=8),
    AnimalRun(mouse="195B", pipeline_id=2, step_id=8),
    AnimalRun(mouse="196B", pipeline_id=2, step_id=8),
    AnimalRun(mouse="198B", pipeline_id=2, step_id=8),
    AnimalRun(mouse="210B", pipeline_id=2, step_id=8),
    AnimalRun(mouse="211B", pipeline_id=2, step_id=8),
    AnimalRun(mouse="212B", pipeline_id=2, step_id=8),
]

OUTPUT_PATH: Path = Path("cohort_dorsoventral.xlsx")


# ================================================================
# 3. Section: Functions
# ================================================================
def analyse_animal(run: AnimalRun) -> dict[str, pd.DataFrame]:
    """Return the per-level and summary dorsal/ventral tables for one animal."""
    project = ClearTissueProject.load(mouse=run.mouse, tissue_type=TISSUE_TYPE)

    batch = project.io.load_batch(pipeline_id=run.pipeline_id, step=run.step_id)
    segments = RawDataLoader(project.source).load_atlas_segments()

    dorsal_mask, ventral_mask, cut = dorsoventral_split(batch.atlas)

    dorsal = count_spinal_level_overlap(batch.cells, batch.atlas, segments, dorsal_mask)
    ventral = count_spinal_level_overlap(
        batch.cells, batch.atlas, segments, ventral_mask
    )

    per_level = _merge_halves(dorsal, ventral)
    summary = _summarise(per_level, cut)

    return {
        "level_dorsoventral": _with_animal(per_level, run.mouse),
        "summary_dorsoventral": _with_animal(summary, run.mouse),
    }


def dorsoventral_split(atlas: Atlas) -> tuple[NDArray, NDArray, float]:
    """Split the atlas into dorsal/ventral halves at its centre of gravity.

    The centre of gravity is the mass-weighted mean index of non-background
    voxels along the dorso-ventral axis (the axis whose orientation letter is
    superior/inferior). Voxels on the superior side of that plane are dorsal,
    the rest ventral.

    Returns:
        ``(dorsal_mask, ventral_mask, cut)`` — two full-shape boolean masks and
        the cut index along the dorso-ventral axis.
    """
    axis, letter = _dorsoventral_axis(atlas)
    mass = atlas.data != _BACKGROUND_LABEL

    other_axes = tuple(a for a in range(mass.ndim) if a != axis)
    profile = mass.sum(axis=other_axes).astype(np.float64)
    if profile.sum() == 0:
        raise ValueError("Atlas has no non-background voxels to split.")

    positions = np.arange(mass.shape[axis])
    cut = float((positions * profile).sum() / profile.sum())

    broadcast_shape = [1, 1, 1]
    broadcast_shape[axis] = mass.shape[axis]
    index = positions.reshape(broadcast_shape)

    # Higher index is superior only when the axis letter is 's'; flip for 'i'.
    superior = index >= cut if letter == _SUPERIOR_LETTER else index < cut

    dorsal_mask = np.broadcast_to(superior, mass.shape)
    ventral_mask = np.broadcast_to(~superior, mass.shape)
    return dorsal_mask, ventral_mask, cut


# ──────────────────────────────────────────────────────
# 3.1 Subsection: Helper Functions
# ──────────────────────────────────────────────────────
def _dorsoventral_axis(atlas: Atlas) -> tuple[int, str]:
    for axis, letter in enumerate(atlas.orientation):
        if letter in _DORSOVENTRAL_LETTERS:
            return axis, letter
    raise ValueError(f"No dorso-ventral axis in orientation '{atlas.orientation}'.")


def _merge_halves(dorsal: pd.DataFrame, ventral: pd.DataFrame) -> pd.DataFrame:
    dorsal_cols = {c: f"{c}_dorsal" for c in _VALUE_COLS}
    ventral_cols = {c: f"{c}_ventral" for c in _VALUE_COLS}

    merged = dorsal.rename(columns=dorsal_cols)[
        ["order", "segment", *dorsal_cols.values()]
    ].merge(
        ventral.rename(columns=ventral_cols)[["order", *ventral_cols.values()]],
        on="order",
        how="left",
    )

    merged.insert(
        2,
        "present",
        (merged["volume_dorsal"] > 0) | (merged["volume_ventral"] > 0),
    )
    return merged


def _summarise(per_level: pd.DataFrame, cut: float) -> pd.DataFrame:
    row: dict[str, float] = {"cut_index": round(cut, 2)}
    for half in ("dorsal", "ventral"):
        cell_sum = int(per_level[f"cell_sum_{half}"].sum())
        volume = int(per_level[f"volume_{half}"].sum())
        row[f"cell_sum_{half}"] = cell_sum
        row[f"volume_{half}"] = volume
        row[f"density_pct_{half}"] = 100.0 * cell_sum / volume if volume else 0.0

    return pd.DataFrame([row])


def _with_animal(frame: pd.DataFrame, mouse: str) -> pd.DataFrame:
    out = frame.copy()
    out.insert(0, "animal", mouse)
    return out


# ================================================================
# 4. Section: Main
# ================================================================
def main() -> None:
    per_sheet: dict[str, list[pd.DataFrame]] = {
        "level_dorsoventral": [],
        "summary_dorsoventral": [],
    }

    for run in ANIMALS:
        tables = analyse_animal(run)
        for sheet, frame in tables.items():
            per_sheet[sheet].append(frame)
        print(f"Analysed {run.mouse} (pipeline {run.pipeline_id}, step {run.step_id})")

    with pd.ExcelWriter(OUTPUT_PATH, engine="openpyxl") as writer:
        for sheet, frames in per_sheet.items():
            combined = pd.concat(frames, ignore_index=True)
            combined.to_excel(writer, sheet_name=sheet, index=False)

    print(f"Wrote {len(ANIMALS)} animals across 2 sheets to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
