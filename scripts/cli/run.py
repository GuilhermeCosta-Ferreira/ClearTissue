# ================================================================
# 0. Section: IMPORTS
# ================================================================
import argparse

from cleartissue.domain_model.data import TissueType
from cleartissue.service.ClearTissueProject import ClearTissueProject
import cleartissue.domain_model.transformations as tr


# ================================================================
# 1. Section: CLI
# ================================================================
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the full transformation pipeline for one mouse.",
    )
    parser.add_argument("mouse_id", help="Mouse identifier, e.g. 189R")
    parser.add_argument(
        "--tissue",
        default="sc",
        help="Tissue type: sc/spine/spinal_cord or br/brain (default: sc)",
    )
    parser.add_argument(
        "--name",
        default="Full Run",
        help="Name for the created pipeline (default: 'Full Run')",
    )
    parser.add_argument(
        "--yes",
        action="store_true",
        help="Skip the interactive config pause before running",
    )
    return parser.parse_args()


# ================================================================
# 2. Section: MAIN
# ================================================================
def main() -> None:
    args = parse_args()

    project = ClearTissueProject.load(
        mouse=args.mouse_id,
        tissue_type=TissueType.from_str(args.tissue),
    )
    raw_batch = project.load_raw()

    pipeline = project.init_pipeline(args.name)

    if not args.yes:
        input("Setup the config. Press enter when ready")

    pipeline.add_list([
        tr.RegularizeSample,
        tr.OrientSample,
        tr.StretchSample,
        tr.StartEndTransformtaion,
        tr.UntwistSample,
        tr.RotateSample,
        tr.CylindricalMaskSample,
        tr.EmptySpaceTrimSample,
        tr.InverseSizeMatchedTissueRegistration,
    ])

    project.run_pipeline(pipeline, raw_batch)


if __name__ == "__main__":
    main()
