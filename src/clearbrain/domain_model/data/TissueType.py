# ================================================================
# 0. Section: IMPORTS
# ================================================================
from enum import Enum


# ================================================================
# 1. Section: Functions
# ================================================================
class TissueType(Enum):
    SPINAL_CORD = 1
    BRAIN = 2

    @classmethod
    def from_str(cls, string: str):
        if string.lower() in ["sc", "spine", "spinal_coord"]:
            return TissueType.SPINAL_CORD
        if string.lower() in ["br", "brain"]:
            return TissueType.BRAIN

        raise NameError(f"{string} is not defined")

    @property
    def as_str(self) -> str:
        if self is TissueType.SPINAL_CORD:
            return "sc"
        if self is TissueType.BRAIN:
            return "br"

        raise ProcessLookupError("Something ain't right")
