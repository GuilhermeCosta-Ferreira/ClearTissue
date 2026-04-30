# ================================================================
# 0. Section: IMPORTS
# ================================================================
import SimpleITK as sitk

from abc import ABC, abstractmethod

from ..configs import RegistrationConfig
from ..methods import RegistrationMethodBuilder



# ================================================================
# 1. Section: Functions
# ================================================================
class RegistratorStrategy(ABC):
    method_builder: RegistrationMethodBuilder = RegistrationMethodBuilder()

    @abstractmethod
    def build_initial_transform(self, fixed: sitk.Image, moving: sitk.Image, config: RegistrationConfig) -> sitk.Transform:
        pass

    def configure(self, fixed: sitk.Image, moving: sitk.Image, config: RegistrationConfig) -> sitk.ImageRegistrationMethod:
        method = self.method_builder.build(config)
        initial_transform = self.build_initial_transform(fixed, moving, config)

        method.SetInitialTransform(initial_transform, inPlace=False)

        return method
