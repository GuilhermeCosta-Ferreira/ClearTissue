# ================================================================
# 0. Section: IMPORTS
# ================================================================
import SimpleITK as sitk

from dataclasses import dataclass

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
    metric_factory: MetricFactory = MetricFactory()
    optimizer_factory: OptimizerFactory = OptimizerFactory()
    interpolator_factory: InterpolationFactory = InterpolationFactory()
    resolution_factory: MultipleResolutionFactory = MultipleResolutionFactory()

    def build(self, config: RegistrationConfig) -> sitk.ImageRegistrationMethod:
        method = sitk.ImageRegistrationMethod()

        self.metric_factory.apply(method, config.metric)
        self.optimizer_factory.apply(method, config.optimizer)
        self.interpolator_factory.apply_registration(method, config.interpolator)
        self.resolution_factory.apply(method, config.multiple_res)

        method.SetOptimizerScalesFromPhysicalShift()

        return method
