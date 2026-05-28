# ================================================================
# 0. Section: IMPORTS
# ================================================================
from dataclasses import dataclass, field

from .config import (
    MetricConfig,
    OptimizerConfig,
    InterpolationConfig,
    MultipleResolutionConfig,
)


# ================================================================
# 1. Section: Functions
# ================================================================
@dataclass
class RegistrationConfig:
    transform_center: int = 1  # 1 is Moments
    metric: MetricConfig = field(default_factory=MetricConfig)
    optimizer: OptimizerConfig = field(default_factory=OptimizerConfig)
    interpolator: InterpolationConfig = field(default_factory=InterpolationConfig)
    multiple_res: MultipleResolutionConfig = field(
        default_factory=MultipleResolutionConfig
    )
