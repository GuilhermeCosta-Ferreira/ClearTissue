# ================================================================
# 0. Section: IMPORTS
# ================================================================
from dataclasses import dataclass, field
from pathlib import Path

from .downloader import download_metadata, download_volume, download_points

from ..tissue import ClearTissue, ClearVolume, TissueType
from .TissueLoader import TissueLoader
from .Metadata import Metadata



# ================================================================
# 1. Section: Functions
# ================================================================
@dataclass(frozen=True, slots=True)
class TissueDownloader:
    mouse: str
    tissue_type: TissueType

    base_path: Path = Path("data/")
    alt_source: str = ""

    _source_filepath: Path = field(init=False, repr=False)

    def __post_init__(self) -> None:
        folder_path = self.base_path / self.mouse

        if len(self.alt_source) <= 0:
            source_filepath = folder_path / f"tissue_{self.tissue_type.str}.json"
        else:
            source_filepath = folder_path / self.alt_source

        object.__setattr__(self, "_source_filepath", source_filepath)


    @classmethod
    def from_loader(cls, loader: TissueLoader) -> 'TissueDownloader':
        return TissueDownloader(
            mouse = loader.mouse,
            tissue_type = loader.tissue_type,
            base_path = loader.base_path,
            alt_source = loader.alt_source
        )


    def download_points(
        self,
        tissue: ClearTissue,
        to_update: bool = False,
        suffix: str = ""
    ) -> Path:
        return download_points(self._source_filepath, tissue, to_update, suffix)

    def download_volume(
        self,
        volume: ClearVolume,
        to_update: bool = False,
        suffix: str = "_volume"
    ) -> Path:
        return download_volume(self._source_filepath, volume, to_update, suffix)

    def download_metadata(
        self,
        metadata: Metadata,
        to_update: bool = False,
        suffix: str = "_metadata"
    ) -> Path:
        return download_metadata(self._source_filepath, metadata, to_update, suffix)
