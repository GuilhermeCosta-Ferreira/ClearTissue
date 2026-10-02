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
        mouse="193R",
        tissue_type=TissueType.SPINAL_CORD,
    )
    raw_batch = project.load_raw()

    pipeline = project.init_pipeline("Evaluation Pipeline")

    input("Setup the config. Press enter when ready")

    pipeline.add_list([
        tr.RegularizeSample,
        tr.OrientSample,
        tr.StretchSample,
    ])

    final_batch = project.run_pipeline(pipeline, raw_batch)
