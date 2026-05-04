# ================================================================
# 0. Section: IMPORTS
# ================================================================
import SimpleITK as sitk

from dataclasses import dataclass, field

from ..configs import RegistrationConfig
from .factories import (
    MetricFactory,
    OptimizerFactory,
    InterpolationFactory,
    MultipleResolutionFactory
)



# ================================================================
# 1. Section: Functions
# ================================================================
@dataclass
class RegistrationMethodBuilder:
    metric_factory: MetricFactory = field(default_factory=MetricFactory)
    optimizer_factory: OptimizerFactory = field(default_factory=OptimizerFactory)
    interpolator_factory: InterpolationFactory = field(default_factory=InterpolationFactory)
    resolution_factory: MultipleResolutionFactory = field(default_factory=MultipleResolutionFactory)

    def build(self, config: RegistrationConfig) -> sitk.ImageRegistrationMethod:
        method = sitk.ImageRegistrationMethod()

        self.metric_factory.apply(method, config.metric)
        self.optimizer_factory.apply(method, config.optimizer)
        self.interpolator_factory.apply_registration(method, config.interpolator)
        self.resolution_factory.apply(method, config.multiple_res)


        return method
