# ================================================================
# 0. Section: IMPORTS
# ================================================================
from clearbrain.domain_model.data import TissueType
from clearbrain.service.ClearTissueProject import ClearTissueProject
from clearbrain.adapters.DataConverter import DataConverter



# ================================================================
# 1. Section: INPUTS
# ================================================================



# ================================================================
# 2. Section: FUNCTIONS
# ================================================================



# ================================================================
# 3. Section: MAIN
# ================================================================
if __name__ == '__main__':
    project = ClearTissueProject.load(
        mouse="32B",
        tissue_type=TissueType.SPINAL_CORD,
    )

    converter = DataConverter(project.source)
    converter.convert_batch(pipeline_id=6, step_id=6)
