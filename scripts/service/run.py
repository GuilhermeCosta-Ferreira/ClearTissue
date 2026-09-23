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
    project = ClearTissueProject.load(
        mouse="01GT-Virus",
        tissue_type=TissueType.SPINAL_CORD,
    )
    raw_batch = project.load_raw()

    pipeline = project.init_pipeline("New registration, this time let's use the tissue template as the fixed image")

    input("Setup the config. Press enter when ready")

    pipeline.add_list([
        tr.RegularizeSample,
        tr.OrientSample,
        #tr.ContrastEnhanceSample,
        tr.StretchSample,
        tr.StartEndTransformtaion,
        #tr.CleanDebrisTransformation,
        tr.UntwistSample,
        tr.RotateSample,
        tr.CylindricalMaskSample,
        tr.EmptySpaceTrimSample,
        tr.InverseSizeMatchedTissueRegistration,
    ])

    final_batch = project.run_pipeline(pipeline, raw_batch)
