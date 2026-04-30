# ================================================================
# 0. Section: IMPORTS
# ================================================================
import time

import numpy as np
import SimpleITK as sitk

from dataclasses import dataclass

from .configs import RegistrationConfig
from .strategies import RegistratorStrategy
from .methods import RegistratorResampler
from .RegistrationResult import RegistrationResult



# ================================================================
# 1. Section: Functions
# ================================================================
@dataclass
class Registrator:
    strategy: RegistratorStrategy
    resampler: RegistratorResampler
    config: RegistrationConfig

    def register(
        self,
        fixed: sitk.Image | np.ndarray,
        moving: sitk.Image | np.ndarray
    ) -> RegistrationResult:
        # 1. Converts the input
        fixed = convert_input(fixed)
        moving = convert_input(moving)

        # 2. Applies the strategy to get the method
        method = self.strategy.configure(fixed, moving, self.config)

        # 3. Applies the registration
        start_time = time.time()
        transform = method.Execute(fixed, moving)
        registration_time = time.time() - start_time

        # 4. Resample the output
        resampled_image = self.apply(fixed, moving, transform)

        # 5. Convert the output to array
        registered_image = sitk.GetArrayFromImage(resampled_image)

        result = RegistrationResult(
            registered_image = registered_image,
            transform = transform,
            final_metric = method.GetMetricValue(),
            stop_condition = method.GetOptimizerStopConditionDescription(),
            elapsed_time = registration_time
        )

        return result

    def apply(self, fixed: sitk.Image | np.ndarray, moving: sitk.Image | np.ndarray, transform: sitk.Transform) -> sitk.Image:
        fixed = convert_input(fixed)
        moving = convert_input(moving)

        resampler = self.resampler.configure(fixed, transform, self.config)
        resampled_image = resampler.Execute(moving)

        return resampled_image


def convert_input(input: sitk.Image | np.ndarray) -> sitk.Image:
    if isinstance(input, np.ndarray):
        return sitk.GetImageFromArray(input.astype(np.float32))
    return input
