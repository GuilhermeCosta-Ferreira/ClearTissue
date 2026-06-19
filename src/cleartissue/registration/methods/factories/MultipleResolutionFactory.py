# ================================================================
# 0. Section: IMPORTS
# ================================================================
import SimpleITK as sitk

from dataclasses import dataclass

from ...configs import MultipleResolutionConfig


# ================================================================
# 1. Section: Functions
# ================================================================
@dataclass
class MultipleResolutionFactory:
    def apply(
        self,
        method: sitk.ImageRegistrationMethod,
        res_config: MultipleResolutionConfig,
    ) -> None:
        if res_config.enabled:
            method.SetShrinkFactorsPerLevel(shrinkFactors=res_config.shrink_factors)
            method.SetSmoothingSigmasPerLevel(
                smoothingSigmas=res_config.smoothing_sigmas
            )
            method.SmoothingSigmasAreSpecifiedInPhysicalUnitsOn()
