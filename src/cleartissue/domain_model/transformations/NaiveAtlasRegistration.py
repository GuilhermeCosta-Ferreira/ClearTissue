# ================================================================
# 0. Section: IMPORTS
# ================================================================
from dataclasses import dataclass

from ..data import SampleBatch
from .utils import register_atlas_to_sample
from .AbstractTransformations import AbstractTransformation
from ...registration.strategies import BSplineRegistration, AffineRegistration
from ...registration import Registrator, RegistrationConfig, RegistratorResampler



# ================================================================
# 1. Section: Functions
# ================================================================
@dataclass
class NaiveAtlasRegistration(AbstractTransformation):
    atlas_affine_registrator_params: dict
    atlas_warp_registrator_params: dict
    max_retries: int

    def __post_init__(self):
        self.affine_registrator_config = RegistrationConfig.from_dict(self.atlas_affine_registrator_params)
        self.warp_registrator_config = RegistrationConfig.from_dict(self.atlas_warp_registrator_params)

        self.affine_registrator = Registrator(
            strategy = AffineRegistration(),
            resampler=RegistratorResampler(),
            config = self.affine_registrator_config,
        )
        self.warp_registrator = Registrator(
            strategy = BSplineRegistration(),
            resampler=RegistratorResampler(),
            config = self.warp_registrator_config,
        )


    def apply(self, batch: SampleBatch) -> SampleBatch:
        registered_atlas = register_atlas_to_sample(
            batch.atlas,
            batch.tissue,
            self.affine_registrator,
            self.warp_registrator,
            self.max_retries,
        )

        return batch.copy_with(atlas=registered_atlas)
