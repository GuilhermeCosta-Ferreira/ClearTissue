# ================================================================
# 0. Section: IMPORTS
# ================================================================
import numpy as np
import SimpleITK as sitk

from dataclasses import dataclass

from .configs import RegistrationConfig
from .strategies import RegistratorStrategy
from .methods import RegistratorResampler
from .RegistrationResult import RegistrationResult
from .QuickRegistrator import QuickRegistrator, convert_input


# ================================================================
# 1. Section: Functions
# ================================================================
@dataclass
class Registrator(QuickRegistrator):
    strategy: RegistratorStrategy
    resampler: RegistratorResampler
    config: RegistrationConfig

    def register(
        self, fixed: sitk.Image | np.ndarray, moving: sitk.Image | np.ndarray
    ) -> RegistrationResult:
        # 1. Applies quick registration (without resampling)
        result = super().register(fixed, moving)

        # 2. Resample the output
        resampled_image = self.apply(fixed, moving, result.transform)

        # 3. Convert the output to array
        registered_image = sitk.GetArrayFromImage(resampled_image)
        result.registered_image = registered_image

        return result

    def apply(
        self,
        fixed: sitk.Image | np.ndarray,
        moving: sitk.Image | np.ndarray,
        transform: sitk.Transform,
    ) -> sitk.Image:
        fixed = convert_input(fixed)
        moving = convert_input(moving)

        resampler = self.resampler.configure(fixed, transform, self.config)
        resampled_image = resampler.Execute(moving)

        return resampled_image
