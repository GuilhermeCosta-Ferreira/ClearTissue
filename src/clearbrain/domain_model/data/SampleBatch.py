# ================================================================
# 0. Section: IMPORTS
# ================================================================
from dataclasses import dataclass

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
