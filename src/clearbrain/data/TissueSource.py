# ================================================================
# 0. Section: IMPORTS
# ================================================================
from dataclasses import dataclass
from pathlib import Path

from ..tissue import TissueType



# ================================================================
# 0. Section: IMPORTS
# ================================================================
@dataclass(frozen=True, slots=True)
class TissueSource:
    mouse: str
    tissue_type: TissueType

    base_path: Path = Path("data/")
    alt_source: str = ""

    @property
    def folder_path(self) -> Path:
        return self.base_path / self.mouse

    @property
    def source_filepath(self) -> Path:
        if self.alt_source:
            return self.folder_path / self.alt_source

        return self.folder_path / f"tissue_{self.tissue_type.str}.json"
