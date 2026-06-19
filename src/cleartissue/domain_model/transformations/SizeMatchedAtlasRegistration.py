# ================================================================
# 0. Section: IMPORTS
# ================================================================
from dataclasses import dataclass

from ..data import SampleBatch
from .AbstractTransformations import AbstractTransformation
from ...registration.strategies import BSplineRegistration, AffineRegistration
from ...registration import Registrator, RegistrationConfig, RegistratorResampler
from .utils import build_size_matched_map, PreferredDirection, register_atlas_to_sample



# ================================================================
# 1. Section: Functions
# ================================================================
@dataclass
class SizeMatchedAtlasRegistration(AbstractTransformation):
    atlas_affine_registrator_params: dict
    atlas_warp_registrator_params: dict
    max_retries: int
    preferred_direction: PreferredDirection

    def __post_init__(self):
        self.affine_registrator_config = RegistrationConfig.from_dict(
            self.atlas_affine_registrator_params
        )
        self.warp_registrator_config = RegistrationConfig.from_dict(
            self.atlas_warp_registrator_params
        )

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
        # 1. Map the axial slices to the size-matched atlas
        atlas_index = build_size_matched_map(batch.tissue, batch.atlas, self.preferred_direction)

        # 2. Apply the affine and warp registrators to the size-matched atlas
        registered_atlas = register_atlas_to_sample(
            batch.atlas, batch.tissue, atlas_index, self.affine_registrator, self.warp_registrator, self.max_retries
        )

        # 3. Return the batch with the registered atlas
        return batch.copy_with(atlas=registered_atlas)
