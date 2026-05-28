# ================================================================
# 0. Section: IMPORTS
# ================================================================
from dataclasses import dataclass

from ..tissue import ClearTissue, ClearVolume
from .loader import load_metadata, load_volume, load_points
from .TissueSource import TissueSource
from .Metadata import Metadata


# ================================================================
# 1. Section: Functions
# ================================================================
@dataclass(frozen=True, slots=True)
class TissueLoader:
    source: TissueSource

    def load_points(self, suffix: str = "") -> ClearTissue:
        return load_points(self.source.source_filepath, suffix)

    def load_volume(
        self, suffix: str = "_volume", sample_factor: int = 25
    ) -> ClearVolume:
        return load_volume(self.source.source_filepath, suffix, sample_factor)

    def load_metadata(self) -> Metadata:
        return load_metadata(self.source.source_filepath)
