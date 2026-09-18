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
        mouse="198B",
        tissue_type=TissueType.SPINAL_CORD,
    )
    raw_batch = project.load_raw()

    pipeline = project.init_pipeline("Contrast enhance method full run")

    input("Setup the config. Press enter when ready")

    pipeline.add_list([
        tr.RegularizeSample,
        tr.OrientSample,
        tr.ContrastEnhanceSample,
        tr.StretchSample,
        tr.StartEndTransformtaion,
        #tr.CleanDebrisTransformation,
        tr.UntwistSample,
        tr.RotateSample,
        tr.CylindricalMaskSample,
        tr.EmptySpaceTrimSample,
        tr.InverseSizeMatchedAtlasRegistration,
    ])

    final_batch = project.run_pipeline(pipeline, raw_batch)
