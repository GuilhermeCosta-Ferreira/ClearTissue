# ================================================================
# 0. Section: IMPORTS
# ================================================================
import pandas as pd

from numpy.typing import NDArray
from dataclasses import dataclass

from .ClearData import ClearData



# ================================================================
# 1. Section: Functions
# ================================================================
@dataclass
class Atlas(ClearData):
    hemisphere: NDArray
    look_up: pd.DataFrame
