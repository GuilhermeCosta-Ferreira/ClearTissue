# ================================================================
# 0. Section: IMPORTS
# ================================================================
from abc import ABC
from dataclasses import dataclass

from ..data import SampleBatch



# ================================================================
# 1. Section: Functions
# ================================================================
@dataclass
class AbstractTransformation(ABC):
    def apply(self, batch: SampleBatch) -> SampleBatch:
        raise NotImplementedError()
