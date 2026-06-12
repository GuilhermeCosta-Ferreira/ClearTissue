# ================================================================
# 0. Section: IMPORTS
# ================================================================
from dataclasses import dataclass, replace

from .Atlas import Atlas
from .ClearVolume import ClearVolume
from .ClearPoints import ClearPoints



# ================================================================
# 1. Section: Functions
# ================================================================
@dataclass
class SampleBatch:
    tissue: ClearVolume
    cells: ClearVolume | ClearPoints
    atlas: Atlas

    def copy_with(
            self,
            *,
            tissue: ClearVolume | None = None,
            cells: ClearVolume | ClearPoints | None = None,
            atlas: Atlas | None = None,
        ) -> "SampleBatch":
            return replace(
                self,
                tissue=self.tissue if tissue is None else tissue,
                cells=self.cells if cells is None else cells,
                atlas=self.atlas if atlas is None else atlas,
            )
