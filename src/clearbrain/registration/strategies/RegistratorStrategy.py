# ================================================================
# 0. Section: IMPORTS
# ================================================================
import SimpleITK as sitk

from abc import ABC, abstractmethod
from dataclasses import field, dataclass

from ..configs import RegistrationConfig
from ..methods import RegistrationMethodBuilder



# ================================================================
# 1. Section: Functions
# ================================================================
@dataclass
class RegistratorStrategy(ABC):
    method_builder: RegistrationMethodBuilder = field(default_factory=RegistrationMethodBuilder)

    @abstractmethod
    def build_initial_transform(self, fixed: sitk.Image, moving: sitk.Image, config: RegistrationConfig) -> sitk.Transform:
        pass

    def configure(self, fixed: sitk.Image, moving: sitk.Image, config: RegistrationConfig) -> sitk.ImageRegistrationMethod:
        method = self.method_builder.build(config)

        initial_transform = self.build_initial_transform(fixed, moving, config)
        if config.optimizer.initial_angle is not None:
            configure_angle(fixed, initial_transform, config.optimizer.initial_angle)

        method.SetInitialTransform(initial_transform, inPlace=False)


        return method

def configure_angle(fixed: sitk.Image, transform: sitk.Transform, angle: float):
    if fixed.GetDimension() == 2:
        transform.SetAngle(angle)
    else:
        raise NotImplementedError(f"No initial angle implementation for dim {fixed.GetDimension()}")

    return transform
