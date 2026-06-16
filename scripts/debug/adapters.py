# ================================================================
# 0. Section: IMPORTS
# ================================================================
from clearbrain.domain_model.data import TissueType
from clearbrain.domain_model.transformations import RegularizeSample
from clearbrain.service.ClearTissueProject import ClearTissueProject



# ================================================================
# 3. Section: MAIN
# ================================================================
if __name__ == '__main__':
    """ project = ClearTissueProject.init(
        mouse="32B",
        tissue_type=TissueType.SPINAL_CORD,
    ) """
    project = ClearTissueProject.load(
        mouse="32B",
        tissue_type=TissueType.SPINAL_CORD,
    )
    raw_batch = project.load_raw()

    pipeline = project.init_pipeline()

    pipeline.add_step(RegularizeSample())

    final_batch = project.run_pipeline(pipeline, raw_batch)
