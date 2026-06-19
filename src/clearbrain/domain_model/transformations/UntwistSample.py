# ================================================================
# 0. Section: IMPORTS
# ================================================================
from dataclasses import dataclass

from ..data import SampleBatch, ClearVolume
from .AbstractTransformations import AbstractTransformation
from .utils import untwist_spinal_coord, apply_know_untwisting
from ...registration import Registrator, RegistrationConfig, RigidRegistration, RegistratorResampler



# ================================================================
# 1. Section: Functions
# ================================================================
@dataclass
class UntwistSample(AbstractTransformation):
    tissue_registrator_params: dict
    cell_registrator_params: dict
    window_size: int
    gap_size: int

    def __post_init__(self):
        self.tissue_registrator_config = RegistrationConfig.from_dict(self.tissue_registrator_params)
        self.cell_registrator_config = self.tissue_registrator_config.with_overrides(self.cell_registrator_params)

        self.tissue_registrator = Registrator(
            strategy = RigidRegistration(),
            resampler=RegistratorResampler(),
            config = self.tissue_registrator_config,
        )
        self.cell_registrator = Registrator(
            strategy = RigidRegistration(),
            resampler=RegistratorResampler(),
            config = self.cell_registrator_config,
        )



    def apply(self, batch: SampleBatch) -> SampleBatch:
        untwisted_tissue, twisting_data = untwist_spinal_coord(
            tissue = batch.tissue,
            registrator = self.tissue_registrator,
            window_size = self.window_size,
            gap = self.gap_size
        )

        if isinstance(batch.cells, ClearVolume):
            untwisted_cell = apply_know_untwisting(
                batch.cells,
                self.cell_registrator,
                twisting_data
            )
        else:
            raise TypeError(
                f"Expected ClearVolume, got {type(batch.cells)} instead"
            )

        return batch.copy_with(tissue=untwisted_tissue, cells=untwisted_cell)
