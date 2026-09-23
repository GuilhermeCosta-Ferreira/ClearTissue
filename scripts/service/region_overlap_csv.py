# ================================================================
# 0. Section: IMPORTS
# ================================================================
from cleartissue.domain_model.data import TissueType
from cleartissue.adapters.RawDataLoader import RawDataLoader
from cleartissue.service.ClearTissueProject import ClearTissueProject
from cleartissue.domain_model.analysis import (
    count_region_overlap,
    count_spinal_level_overlap,
    region_spinal_levels,
)

# ================================================================
# 1. Section: INPUTS
# ================================================================
MOUSE: str = "01GT-Virus"
TISSUE_TYPE: TissueType = TissueType.SPINAL_CORD

PIPELINE_ID: int = 10
STEP_ID: int = 8


# ================================================================
# 2. Section: MAIN
# ================================================================
if __name__ == "__main__":
    project = ClearTissueProject.load(mouse=MOUSE, tissue_type=TISSUE_TYPE)

    batch = project.io.load_batch(pipeline_id=PIPELINE_ID, step=STEP_ID)
    raw_loader = RawDataLoader(project.source)
    structures = raw_loader.load_atlas_structures()
    segments = raw_loader.load_atlas_segments()

    overlap = count_region_overlap(batch.cells, batch.atlas, structures)
    levels = region_spinal_levels(batch.atlas, segments, structures)
    overlap["spinal_levels"] = overlap["id"].map(
        lambda region_id: ";".join(levels.get(region_id, []))
    )

    leaves = overlap[overlap["children"] == ""]  # leaf regions only, no parents
    by_level = count_spinal_level_overlap(batch.cells, batch.atlas, segments)

    step_path = project.source.step_path(PIPELINE_ID, STEP_ID)
    base = project.source.file_base_name

    out_path = step_path / f"{base}_region_overlap.csv"
    leaves_path = step_path / f"{base}_region_overlap_leaves.csv"
    level_path = step_path / f"{base}_spinal_level_overlap.csv"

    overlap.to_csv(out_path, index=False)
    leaves.to_csv(leaves_path, index=False)
    by_level.to_csv(level_path, index=False)

    print(f"Wrote {len(overlap)} regions to {out_path}")
    print(f"Wrote {len(leaves)} leaf regions to {leaves_path}")
    print(f"Wrote {len(by_level)} spinal levels to {level_path}")
