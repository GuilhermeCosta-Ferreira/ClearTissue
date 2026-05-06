# ================================================================
# 0. Section: IMPORTS
# ================================================================
from dataclasses import dataclass
from pathlib import Path

from ..registration import RegistrationResult
from ..tissue import ClearTissue, ClearVolume
from .downloader import (
    download_metadata,
    download_volume,
    download_points,
    download_twisting_data
)
from .TissueSource import TissueSource
from .Metadata import Metadata



# ================================================================
# 1. Section: Functions
# ================================================================
@dataclass(frozen=True, slots=True)
class TissueDownloader:
    source: TissueSource


    def download_points(
        self,
        tissue: ClearTissue,
        to_update: bool = False,
        suffix: str = ""
    ) -> Path:
        return download_points(
            self.source.source_filepath,
            tissue,
            to_update,
            suffix
        )

    def download_volume(
        self,
        volume: ClearVolume,
        to_update: bool = False,
        suffix: str = "_volume"
    ) -> Path:
        return download_volume(
            self.source.source_filepath,
            volume,
            to_update,
            suffix
        )

    def download_metadata(
        self,
        metadata: Metadata,
        to_update: bool = False,
        suffix: str = "_metadata"
    ) -> Path:
        return download_metadata(
            self.source.source_filepath,
            metadata,
            to_update,
            suffix
        )

    def download_twisting_data(
        self,
        twisting_data: list[RegistrationResult],
        to_update: bool = False,
        suffix: str = "_twisting_data"
    ) -> Path:
        return download_twisting_data(
            self.source.source_filepath,
            twisting_data,
            to_update,
            suffix
        )
