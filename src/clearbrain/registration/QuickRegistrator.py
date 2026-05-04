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
from .Registrator import convert_input



# ================================================================
# 1. Section: Functions
# ================================================================
@dataclass
class QuickRegistrator:
    strategy: RegistratorStrategy
    resampler: RegistratorResampler
    config: RegistrationConfig

    def quick_registration(
        self,
        fixed: sitk.Image | np.ndarray,
        moving: sitk.Image | np.ndarray
    ) -> RegistrationResult:
        """No resampling applied, registered image is empty"""
        # 1. Converts the input
        fixed = convert_input(fixed)
        moving = convert_input(moving)

        # 2. Applies the strategy to get the method
        method = self.strategy.configure(fixed, moving, self.config)

        # 3. Applies the registration
        start_time = time.time()
        transform = method.Execute(fixed, moving)
        registration_time = time.time() - start_time

        return RegistrationResult(
            registered_image = np.zeros_like(moving),
            transform = transform,
            final_metric = method.GetMetricValue(),
            stop_condition = method.GetOptimizerStopConditionDescription(),
            elapsed_time = registration_time
        )

    def quick_apply(
        self,
        fixed: sitk.Image | np.ndarray,
        moving: sitk.Image | np.ndarray,
        result: RegistrationResult,
    ) -> RegistrationResult:
        """Used after a quick registration"""
        fixed = convert_input(fixed)
        moving = convert_input(moving)

        resampler = self.resampler.configure(fixed, result.transform, self.config)
        resampled_image = resampler.Execute(moving)

        result.registered_image = resampled_image

        return result
