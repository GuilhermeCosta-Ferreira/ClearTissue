# ================================================================
# 0. Section: IMPORTS
# ================================================================
from dataclasses import dataclass, field
from pathlib import Path


from .loader import load_metadata, load_volume, load_points

from ..tissue import ClearTissue, ClearVolume, TissueType
from .TissueDownloader import TissueDownloader
from .Metadata import Metadata



# ================================================================
# 1. Section: Functions
# ================================================================
@dataclass(frozen=True, slots=True)
class TissueLoader:
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
    def from_downloader(cls, downloader: TissueDownloader) -> 'TissueLoader':
        return TissueLoader(
            mouse = downloader.mouse,
            tissue_type = downloader.tissue_type,
            base_path = downloader.base_path,
            alt_source = downloader.alt_source
        )




    def load_points(self) -> ClearTissue:
        return load_points(self._source_filepath)

    def load_volume(self) -> ClearVolume:
        return load_volume(self._source_filepath)

    def load_metadata(self) -> Metadata:
        return load_metadata(self._source_filepath)
