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


    def load_points(self) -> ClearTissue:
        return load_points(self.source.source_filepath)

    def load_volume(self) -> ClearVolume:
        return load_volume(self.source.source_filepath)

    def load_metadata(self) -> Metadata:
        return load_metadata(self.source.source_filepath)
