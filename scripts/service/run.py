# ================================================================
# 0. Section: IMPORTS
# ================================================================
from cleartissue.domain_model.data import TissueType
from cleartissue.service.ClearTissueProject import ClearTissueProject
import cleartissue.domain_model.transformations as tr



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

    pipeline = project.init_pipeline("Cell Counts")

    input("Setup the config. Press enter when ready")

    pipeline.add_list([
        tr.RegularizeSample,
        tr.OrientSample,
        tr.StretchSample,
        tr.UntwistSample,
        tr.RotateSample,
        tr.CylindricalMaskSample,
        tr.EmptySpaceTrimSample,
        tr.SizeMatchedAtlasRegistration,
        tr.PruneAtlas
    ])

    final_batch = project.run_pipeline(pipeline, raw_batch)
