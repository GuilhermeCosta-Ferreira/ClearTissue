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
class RigidRegistration(RegistratorStrategy):
    def build_initial_transform(
        self,
        fixed: sitk.Image,
        moving: sitk.Image,
        config: RegistrationConfig,
    ) -> sitk.Transform:
        initial_transform = sitk.CenteredTransformInitializer(
            fixed,
            moving,
            get_dim(fixed),
            config.transform_center
        )

        return initial_transform

def get_dim(image: sitk.Image) -> sitk.Transform:
    dim = image.GetDimension()

    if dim == 2:
        transform = sitk.Euler2DTransform()
    elif dim == 3:
        transform = sitk.Euler3DTransform()
    else:
        raise ValueError(f"Unsupported image dimension for rigid registration: {dim}")

    return transform
