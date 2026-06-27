# ================================================================
# 0. Section: IMPORTS
# ================================================================
from dataclasses import dataclass

from ..data import SampleBatch, ClearPoints
from .AbstractTransformations import AbstractTransformation
from ...registration.strategies import BSplineRegistration, AffineRegistration
from ...registration import Registrator, RegistrationConfig, RegistratorResampler
from .utils import build_size_matched_map, PreferredDirection, register_sample_to_atlas



# ================================================================
# 1. Section: Functions
# ================================================================
@dataclass
class InverseSizeMatchedAtlasRegistration(AbstractTransformation):
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

        if isinstance(batch.cells, ClearPoints):
            raise TypeError(
                f"Expected ClearVolume, got {type(batch.cells)} for the cells,"
                " make sure you run RegularizeSample first"
            )

        # 2. Apply the affine and warp registrators to the size-matched atlas
        registered_tissue, registered_cells = register_sample_to_atlas(
            batch.atlas, batch.tissue, batch.cells, atlas_index, self.affine_registrator, self.warp_registrator, self.max_retries
        )

        # 3. Return the batch with the registered atlas
        return batch.copy_with(tissue=registered_tissue, cells=registered_cells)
