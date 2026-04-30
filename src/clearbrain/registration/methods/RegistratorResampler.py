# ================================================================
# 0. Section: IMPORTS
# ================================================================
import SimpleITK as sitk

from dataclasses import dataclass

from .factories import InterpolationFactory
from ..configs import RegistrationConfig



# ================================================================
# 1. Section: Functions
# ================================================================
@dataclass
class RegistratorResampler:
    interpolation_factory: InterpolationFactory = InterpolationFactory()

    def configure(
        self,
        fixed: sitk.Image,
        transform: sitk.Transform,
        config: RegistrationConfig
    ) -> sitk.ResampleImageFilter:
        resampler = sitk.ResampleImageFilter()
        resampler.SetReferenceImage(fixed)

        self.interpolation_factory.apply_resampling(resampler, config.interpolator)
        resampler.SetTransform(transform)

        return resampler
