# ================================================================
# 0. Section: IMPORTS
# ================================================================
import SimpleITK

from typing import Type, Any

from ..domain_model.data import SampleBatch
from ..domain_model.execution_metadata import StepSignature
from ..domain_model.transformations import AbstractTransformation
from .hash import hash_transformation_code, hash_sample_batch, stable_hash



# ================================================================
# 1. Section: Functions
# ================================================================
class StepSignatureBuilder:
    def build(
        self,
        *,
        step_class: Type[AbstractTransformation],
        params: dict[str, Any],
        input_batch: SampleBatch,
    ) -> StepSignature:
        step_name = step_class.__name__

        params_hash = stable_hash(params)
        input_hash = hash_sample_batch(input_batch)
        code_hash = hash_transformation_code(step_class)
        simple_itk_version = str(SimpleITK.SITK_ITK_VERSION_MAJOR) + \
            "." + \
            str(SimpleITK.SITK_ITK_VERSION_MINOR)

        return StepSignature(
            step_name=step_name,
            params_hash=params_hash,
            input_hash=input_hash,
            code_hash=code_hash,
            SimpleITK_version=simple_itk_version,
        )
