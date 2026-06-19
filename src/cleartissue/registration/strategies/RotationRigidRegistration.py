# ================================================================
# 0. Section: IMPORTS
# ================================================================
import SimpleITK as sitk

from dataclasses import dataclass

from .RigidRegistration import RigidRegistration, get_dim
from ..configs import RegistrationConfig


# ================================================================
# 1. Section: Functions
# ================================================================
@dataclass
class RotationRigidRegistration(RigidRegistration):
    def configure(
        self, fixed: sitk.Image, moving: sitk.Image, config: RegistrationConfig
    ) -> sitk.ImageRegistrationMethod:
        method = super().configure(fixed, moving, config)

        if fixed.GetDimension() == 2:
            method.SetOptimizerWeights([1.0, 0.0, 0.0])
        elif fixed.GetDimension() == 3:
            method.SetOptimizerWeights([1, 1, 1, 0, 0, 0])

        return method

    def build_initial_transform(
        self,
        fixed: sitk.Image,
        moving: sitk.Image,
        config: RegistrationConfig,
    ) -> sitk.Transform:
        initial_transform = sitk.CenteredTransformInitializer(
            fixed, moving, get_dim(fixed), config.transform_center
        )

        if fixed.GetDimension() == 2:
            initial_transform.SetTranslation((0.0, 0.0))
        elif fixed.GetDimension() == 3:
            initial_transform.SetTranslation((0.0, 0.0, 0.0))

        return initial_transform
