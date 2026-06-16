# ================================================================
# 0. Section: IMPORTS
# ================================================================
from clearbrain.adapters import Source, Repository
from clearbrain.domain_model.data import TissueType



# ================================================================
# 3. Section: MAIN
# ================================================================
if __name__ == '__main__':
    source = Source(
        mouse="32B",
        tissue_type=TissueType.SPINAL_CORD,
    )

    repository = Repository(source)
    repository.init_project()
    repository.init_new_pipeline(pipeline_id=1, pipeline_name="test_pipeline")
