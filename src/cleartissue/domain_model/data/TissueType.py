# ================================================================
# 0. Section: IMPORTS
# ================================================================
from enum import Enum


# ================================================================
# 1. Section: Functions
# ================================================================
class TissueType(Enum):
    SPINAL_CORD = "sc"
    BRAIN = "br"

    @classmethod
    def from_str(cls, string: str):
        if string.lower() in ["sc", "spine", "spinal_cord"]:
            return TissueType.SPINAL_CORD
        if string.lower() in ["br", "brain"]:
            return TissueType.BRAIN

        raise NameError(f"{string} is not defined")

    @property
    def as_str(self) -> str:
        return self.value
