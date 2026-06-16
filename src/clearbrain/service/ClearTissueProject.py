# ================================================================
# 0. Section: IMPORTS
# ================================================================
from pathlib import Path
from brainglobe_atlasapi.atlas_name import AtlasName

from ..adapters.Source import Source
from ..adapters.ClearIO import ClearIO
from .PipelineSpecs import PipelineSpecs
from .PipelineRunner import PipelineRunner
from ..adapters.Repository import Repository
from ..domain_model.data import TissueType, SampleBatch



# ================================================================
# 1. Section: Functions
# ================================================================
class ClearTissueProject:
    def __init__(
        self,
        source: Source,
        clear_io: ClearIO,
        runner: PipelineRunner,
    ):
        self.source = source
        self.io = clear_io
        self.runner = runner



    # ================================================================
    # 2. Section: Class Methods
    # ================================================================
    @classmethod
    def init(
        cls,
        mouse: str,
        tissue_type: TissueType,
        folder_path: Path = Path("data"),
        atlas_name: AtlasName = "allen_cord_20um",
    ) -> "ClearTissueProject":
        source = Source(
            mouse=mouse,
            tissue_type=tissue_type,
            base_path=folder_path,
            atlas_name=atlas_name,
        )

        repository = Repository(source=source)
        repository.init_project()

        clear_io = ClearIO(source=source)
        runner = PipelineRunner(io=clear_io)

        return cls(source=source, clear_io=clear_io, runner=runner)

    @classmethod
    def load(
        cls,
        mouse: str,
        tissue_type: TissueType,
        folder_path: Path = Path("data"),
        atlas_name: AtlasName = "allen_cord_20um",
    ) -> "ClearTissueProject":
        source = Source(
            mouse=mouse,
            tissue_type=tissue_type,
            base_path=folder_path,
            atlas_name=atlas_name,
        )
        clear_io = ClearIO(source=source)
        runner = PipelineRunner(io=clear_io)

        return cls(source=source, clear_io=clear_io, runner=runner)



    # ================================================================
    # 3. Section: Methods
    # ================================================================
    def load_raw(self) -> SampleBatch:
        return self.io.load_raw()

    def init_pipeline(self, pipeline_name: str = "") -> PipelineSpecs:
        repository = Repository(source=self.source)
        pipeline_id = repository.get_available_id()

        pipeline_name = repository.init_new_pipeline(pipeline_name=pipeline_name, pipeline_id=pipeline_id)
        return PipelineSpecs(pipeline_name=pipeline_name, pipeline_id=pipeline_id)

    def run_pipeline(self, pipeline: PipelineSpecs, batch: SampleBatch, save_intermediates: bool = True) -> SampleBatch:
        return self.runner.run(
            batch=batch,
            pipeline=pipeline,
            save_intermediates=save_intermediates,
        )
