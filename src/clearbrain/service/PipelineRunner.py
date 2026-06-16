# ================================================================
# 0. Section: IMPORTS
# ================================================================
from dataclasses import dataclass

from .PipelineSpecs import PipelineSpecs
from ..domain_model.data import SampleBatch
from ..adapters.ClearIO import ClearIO



# ================================================================
# 1. Section: Functions
# ================================================================
@dataclass
class PipelineRunner:
    io: ClearIO

    def run(self, batch: SampleBatch, pipeline: PipelineSpecs, save_intermediates: bool) -> SampleBatch:
        for i, step in enumerate(pipeline.steps):
            print(f"Running step {i}: {step.__class__.__name__}")
            batch = step.apply(batch)

            if save_intermediates:
                self.io.download_batch(batch, pipeline.pipeline_id, i)

        if not save_intermediates:
            self.io.download_batch(batch, pipeline.pipeline_id, len(pipeline.steps) - 1)

        return batch
