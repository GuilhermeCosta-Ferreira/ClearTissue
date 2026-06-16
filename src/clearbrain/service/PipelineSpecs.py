# ================================================================
# 0. Section: IMPORTS
# ================================================================
from dataclasses import dataclass, field

from ..domain_model.transformations import AbstractTransformations



# ================================================================
# 1. Section: Functions
# ================================================================
@dataclass
class PipelineSpecs:
    pipeline_id: int
    pipeline_name: str
    steps: list[AbstractTransformations] = field(default_factory=list)

    def add_list(self, steps: list[AbstractTransformations]) -> None:
        self.steps.extend(steps)

    def add_step(self, step: AbstractTransformations) -> None:
        self.steps.append(step)

    def insert_step(self, index: int, step: AbstractTransformations) -> None:
        self.steps.insert(index, step)

    def remove_step(self, index: int) -> None:
        self.steps.pop(index)

    def clear(self) -> None:
        self.steps.clear()
