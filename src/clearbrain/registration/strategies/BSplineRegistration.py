# ================================================================
# 0. Section: IMPORTS
# ================================================================
import SimpleITK as sitk

from dataclasses import dataclass

from .RegistratorStrategy import RegistratorStrategy
from ..configs import RegistrationConfig


# ================================================================
# 1. Section: Functions
# ================================================================
@dataclass
class BSplineRegistration(RegistratorStrategy):
    def build_initial_transform(
        self,
        fixed: sitk.Image,
        moving: sitk.Image,
        config: RegistrationConfig,
    ) -> sitk.Transform:
        return sitk.BSplineTransformInitializer(
            fixed, [config.optimizer.grid_size] * fixed.GetDimension()
        )
