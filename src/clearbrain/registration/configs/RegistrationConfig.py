# ================================================================
# 0. Section: IMPORTS
# ================================================================
from dataclasses import dataclass

from .config import (
    MetricConfig,
    OptimizerConfig,
    InterpolationConfig,
    MultipleResolutionConfig
)

# ================================================================
# 1. Section: Functions
# ================================================================
@dataclass
class RegistrationConfig:
    transform_center: int # 1 is Moments
    metric: MetricConfig
    optimizer: OptimizerConfig
    interpolator: InterpolationConfig
    multiple_res: MultipleResolutionConfig
