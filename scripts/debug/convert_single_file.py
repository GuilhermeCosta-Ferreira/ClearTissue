# ================================================================
# 0. Section: IMPORTS
# ================================================================
from cleartissue import ClearTissueProject
from cleartissue.domain_model import TissueType
from cleartissue.adapters.DataConverter import DataConverter



# ================================================================
# 1. Section: INPUTS
# ================================================================
MOUSE: str = "198B-Cells"
PIPELINE_ID: int = -1
STEP: int = 0


# ================================================================
# 2. Section: FUNCTIONS
# ================================================================



# ================================================================
# 3. Section: MAIN
# ================================================================
if __name__ == '__main__':
    project = ClearTissueProject.load(
        mouse=MOUSE,
        tissue_type=TissueType.SPINAL_CORD,
    )
    converter = DataConverter(project.source)

    tissue = project.io._loader._step_loader.load_tissue(PIPELINE_ID, STEP)
    converter.nii_converter._convert_volume(tissue, project.source.step_path(PIPELINE_ID, STEP))
