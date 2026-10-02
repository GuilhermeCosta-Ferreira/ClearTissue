# ================================================================
# 0. Section: IMPORTS
# ================================================================
import argparse

from cleartissue.domain_model.data import TissueType
from cleartissue.service.ClearTissueProject import ClearTissueProject
from cleartissue.adapters.DataConverter import DataConverter


# ================================================================
# 1. Section: CLI
# ================================================================
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert a pipeline step's batch to another file type.",
    )
    parser.add_argument("mouse_id", help="Mouse identifier, e.g. 189R")
    parser.add_argument("pipeline_id", type=int, help="Pipeline identifier")
    parser.add_argument("step_id", type=int, help="Step identifier")
    parser.add_argument(
        "--tissue",
        default="sc",
        help="Tissue type: sc/spine/spinal_cord or br/brain (default: sc)",
    )
    parser.add_argument(
        "--out-type",
        default=".nii.gz",
        help="Output file type (default: .nii.gz)",
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

    converter = DataConverter(project.source)
    converter.convert_batch(
        pipeline_id=args.pipeline_id,
        step_id=args.step_id,
        out_file_type=args.out_type,
    )


if __name__ == "__main__":
    main()
