# ================================================================
# 0. Section: IMPORTS
# ================================================================
import numpy as np

from dataclasses import dataclass
from ..meta import Metadata



# ================================================================
# 1. Section: Functions
# ================================================================
@dataclass
class ClearTissue:
    points: np.ndarray
    metadata: Metadata
