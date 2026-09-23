# ================================================================
# 0. Section: IMPORTS
# ================================================================
from cleartissue.domain_model.data import TissueType
from cleartissue.service.ClearTissueProject import ClearTissueProject
from cleartissue.adapters.DataConverter import DataConverter



# ================================================================
# 1. Section: MAIN
# ================================================================
if __name__ == '__main__':
    project = ClearTissueProject.load(
        mouse="198B-Cells",
        tissue_type=TissueType.SPINAL_CORD,
    )

    converter = DataConverter(project.source)
    converter.convert_batch(pipeline_id=-1, step_id=0, out_file_type=".nii.gz")
