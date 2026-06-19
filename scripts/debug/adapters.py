# ================================================================
# 0. Section: IMPORTS
# ================================================================
from clearbrain.domain_model.data import TissueType
from clearbrain.service.ClearTissueProject import ClearTissueProject
import clearbrain.domain_model.transformations as tr




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

    pipeline = project.init_pipeline("Cache versions")

    input("Setup the config. Press enter when ready")

    pipeline.add_list([
        tr.RegularizeSample,
        tr.OrientSample,
        tr.StretchSample,
        tr.UntwistSample,
        tr.RotateSample,
        tr.CylindricalMaskSample,
        tr.EmptySpaceTrimSample,
    ])

    final_batch = project.run_pipeline(pipeline, raw_batch)
