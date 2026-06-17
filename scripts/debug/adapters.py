# ================================================================
# 0. Section: IMPORTS
# ================================================================
from clearbrain.domain_model.data import TissueType
from clearbrain.service.ClearTissueProject import ClearTissueProject
from clearbrain.domain_model.transformations import RegularizeSample, OrientSample



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

    pipeline.add_list([
        RegularizeSample(),
        OrientSample(),
    ])

    final_batch = project.run_pipeline(pipeline, raw_batch)
