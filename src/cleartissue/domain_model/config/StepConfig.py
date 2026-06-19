# ================================================================
# 0. Section: IMPORTS
# ================================================================
from dataclasses import dataclass
from typing import Any



# ================================================================
# 1. Section: Functions
# ================================================================
@dataclass(frozen=True)
class StepConfig:
    name: str
    enabled: bool
    parameters: dict[str, Any]
