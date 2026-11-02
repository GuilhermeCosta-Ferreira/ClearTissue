# ================================================================
# 0. Section: IMPORTS
# ================================================================
import pandas as pd

from pathlib import Path
from dataclasses import dataclass

from cleartissue.domain_model.data import TissueType
from cleartissue.adapters.RawDataLoader import RawDataLoader
from cleartissue.service.ClearTissueProject import ClearTissueProject
from cleartissue.domain_model.analysis import (
    count_region_overlap_by_hemisphere,
    count_spinal_level_overlap_by_hemisphere,
    region_spinal_levels,
)


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

OUTPUT_PATH: Path = Path("cohort_analysis.xlsx")


# ================================================================
# 3. Section: Functions
# ================================================================
def analyse_animal(run: AnimalRun) -> dict[str, pd.DataFrame]:
    project = ClearTissueProject.load(mouse=run.mouse, tissue_type=TISSUE_TYPE)

    batch = project.io.load_batch(pipeline_id=run.pipeline_id, step=run.step_id)
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
        "region_overlap": _with_animal(overlap, run.mouse),
        "region_overlap_leaves": _with_animal(leaves, run.mouse),
        "spinal_level_overlap": _with_animal(by_level, run.mouse),
    }


# ──────────────────────────────────────────────────────
# 3.1 Subsection: Helper Functions
# ──────────────────────────────────────────────────────
def _with_animal(frame: pd.DataFrame, mouse: str) -> pd.DataFrame:
    out = frame.copy()
    out.insert(0, "animal", mouse)
    return out


# ================================================================
# 4. Section: Main
# ================================================================
def main() -> None:
    per_sheet: dict[str, list[pd.DataFrame]] = {
        "region_overlap": [],
        "region_overlap_leaves": [],
        "spinal_level_overlap": [],
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

    print(f"Wrote {len(ANIMALS)} animals across 3 sheets to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
