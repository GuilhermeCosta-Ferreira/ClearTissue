# ================================================================
# 0. Section: IMPORTS
# ================================================================
from dataclasses import dataclass
from pathlib import Path



from .Source import Source
from .DataLoader import DataLoader
from .DataDownloader import DataDownloader
from ..domain_model.data import SampleBatch
from .StepManifestRepository import StepManifestRepository
from .PipelineConfigRepository import PipelineConfigRepository
from ..domain_model.execution_metadata import StepSignature, StepManifest, StepStatus



# ================================================================
# 1. Section: Functions
# ================================================================
@dataclass
class ClearIO:
    source: Source

    def __post_init__(self):
        self._loader = DataLoader(self.source)
        self._downloader = DataDownloader(self.source)
        self._pipeline_repo = PipelineConfigRepository(self.source)
        self._step_manifest_repo = StepManifestRepository(self.source)

    def load_raw(self) -> SampleBatch:
        return self._loader.load_raw()

    def load_batch(self, pipeline_id: int, step: int) -> SampleBatch:
        return self._loader.load_batch(pipeline_id, step)

    def download_batch(self, batch: SampleBatch, pipeline_id: int, step: int) -> Path:
        return self._downloader.download_batch(batch, pipeline_id, step)

    def load_pipeline_config(self, pipeline_id: int):
        return self._pipeline_repo.load_config(pipeline_id=pipeline_id)

    def find_reusable_step(self, signature: StepSignature) -> StepManifest | None:
        return self._step_manifest_repo.find_by_signature(signature.value)

    def save_redirect(self, pipeline_id: int, step_id: int, source_pipeline_id: int, source_step_id: int, signature: str) -> None:
        self._step_manifest_repo.save_redirect(pipeline_id, step_id, source_pipeline_id, source_step_id, signature)

    def save_step_manifest(self,
        pipeline_id: int,
        step_id: int,
        step_name: str,
        signature: str,
        status: str
    ) -> None:

        manifest = StepManifest(
            pipeline_id=pipeline_id,
            step_id=step_id,
            step_name=step_name,
            signature=signature,
            status=StepStatus(status),
        )
        self._step_manifest_repo.save_manifest(manifest)
