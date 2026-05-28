# ================================================================
# 0. Section: IMPORTS
# ================================================================
from enum import Enum


# ================================================================
# 1. Section: Functions
# ================================================================
class TissueType(Enum):
    SPINAL_COORD = 1
    BRAIN = 2

    @classmethod
    def from_str(cls, string: str):
        if string.lower() in ["sc", "spine", "spinal_coord"]:
            return TissueType.SPINAL_COORD
        if string.lower() in ["br", "brain"]:
            return TissueType.BRAIN

        raise NameError(f"{string} is not defined")

    @property
    def as_str(self) -> str:
        if TissueType.SPINAL_COORD:
            return "sc"
        if TissueType.BRAIN:
            return "br"

        raise ProcessLookupError("Something ain't right")
