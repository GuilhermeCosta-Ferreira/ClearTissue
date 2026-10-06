# ================================================================
# 0. Section: IMPORTS
# ================================================================
from pathlib import Path

import pandas as pd

from cleartissue.domain_model.data import TissueType
from cleartissue.adapters.RawDataLoader import RawDataLoader
from cleartissue.service.ClearTissueProject import ClearTissueProject
from cleartissue.domain_model.analysis import (
    count_region_overlap_by_hemisphere,
    count_spinal_level_overlap_by_hemisphere,
    region_spinal_levels,
)


# ================================================================
# 1. Section: Inputs
# ================================================================
# Edit these in place — this is a cohort script, not a CLI.
ANIMALS: list[str] = ["189R", "193R", "212B"]
TISSUE_TYPE: TissueType = TissueType.SPINAL_CORD

PIPELINE_ID: int = 2
STEP_ID: int = 8

OUTPUT_PATH: Path = Path("cohort_analysis.xlsx")


# ================================================================
# 2. Section: Functions
# ================================================================
def analyse_animal(mouse: str) -> dict[str, pd.DataFrame]:
    """Compute the per-animal overlap tables for one mouse.

    Mirrors analyse.py but returns the frames (whole/left/right hemisphere)
    with a leading ``animal`` column instead of writing per-step CSVs.
    """
    project = ClearTissueProject.load(mouse=mouse, tissue_type=TISSUE_TYPE)

    batch = project.io.load_batch(pipeline_id=PIPELINE_ID, step=STEP_ID)
    raw_loader = RawDataLoader(project.source)
    structures = raw_loader.load_atlas_structures()
    segments = raw_loader.load_atlas_segments()

    overlap = count_region_overlap_by_hemisphere(batch.cells, batch.atlas, structures)
    levels = region_spinal_levels(batch.atlas, segments, structures)
    overlap["spinal_levels"] = overlap["id"].map(
        lambda region_id: ";".join(levels.get(region_id, []))
    )

    leaves = overlap[overlap["children"] == ""]  # leaf regions only, no parents
    by_level = count_spinal_level_overlap_by_hemisphere(
        batch.cells, batch.atlas, segments
    )

    return {
        "region_overlap": _with_animal(overlap, mouse),
        "region_overlap_leaves": _with_animal(leaves, mouse),
        "spinal_level_overlap": _with_animal(by_level, mouse),
    }


# ──────────────────────────────────────────────────────
# 2.1 Subsection: Helper Functions
# ──────────────────────────────────────────────────────
def _with_animal(frame: pd.DataFrame, mouse: str) -> pd.DataFrame:
    out = frame.copy()
    out.insert(0, "animal", mouse)
    return out


# ================================================================
# 3. Section: Main
# ================================================================
def main() -> None:
    per_sheet: dict[str, list[pd.DataFrame]] = {
        "region_overlap": [],
        "region_overlap_leaves": [],
        "spinal_level_overlap": [],
    }

    for mouse in ANIMALS:
        tables = analyse_animal(mouse)
        for sheet, frame in tables.items():
            per_sheet[sheet].append(frame)
        print(f"Analysed {mouse}")

    with pd.ExcelWriter(OUTPUT_PATH, engine="openpyxl") as writer:
        for sheet, frames in per_sheet.items():
            combined = pd.concat(frames, ignore_index=True)
            combined.to_excel(writer, sheet_name=sheet, index=False)

    print(f"Wrote {len(ANIMALS)} animals across 3 sheets to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
