# ================================================================
# 0. Section: IMPORTS
# ================================================================
from dataclasses import dataclass

from ..adapters.ClearIO import ClearIO
from .PipelineSpecs import PipelineSpecs
from ..domain_model.data import SampleBatch
from ..domain_model.config import PipelineConfig
from .StepSignatureBuilder import StepSignatureBuilder



# ================================================================
# 1. Section: Functions
# ================================================================
@dataclass
class PipelineRunner:
    io: ClearIO

    def __post_init__(self):
        self.signature_builder = StepSignatureBuilder()

    def run(
        self,
        batch: SampleBatch,
        pipeline: PipelineSpecs,
        config: PipelineConfig,
        save_intermediates: bool
    ) -> SampleBatch:
        for i, step_class in enumerate(pipeline.steps):
            print(f"Running step {i}: {step_class.__name__}")

            params = config.step_params(step_class.__name__)
            signature = self.signature_builder.build(
                step_class=step_class,
                params=params,
                input_batch=batch,
            )

            reusable = self.io.find_reusable_step(signature)

            if reusable is not None:
                self.io.save_redirect(
                    pipeline_id=pipeline.pipeline_id,
                    step_id=i,
                    source_pipeline_id=reusable.pipeline_id,
                    source_step_id=reusable.step_id,
                    signature=signature.value,
                )
                batch = self.io.load_batch(reusable.pipeline_id, reusable.step_id)
                continue

            batch = step_class(**params).apply(batch)

            if save_intermediates:
                self.io.download_batch(batch, pipeline.pipeline_id, i)
                self.io.save_step_manifest(
                    pipeline_id=pipeline.pipeline_id,
                    step_id=i,
                    step_name=step_class.__name__,
                    signature=signature.value,
                    status="materialized",
                )

        if not save_intermediates:
            self.io.download_batch(batch, pipeline.pipeline_id, len(pipeline.steps) - 1)

        return batch
