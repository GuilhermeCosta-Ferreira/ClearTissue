# ================================================================
# 0. Section: IMPORTS
# ================================================================
from typing import Type
from dataclasses import dataclass, field

from ..domain_model.transformations import AbstractTransformation



# ================================================================
# 1. Section: Functions
# ================================================================
@dataclass
class PipelineSpecs:
    pipeline_id: int
    pipeline_name: str

    steps: list[Type[AbstractTransformation]] = field(default_factory=list)

    def add_list(self, steps: list[Type[AbstractTransformation]]) -> None:
        self.steps.extend(steps)

    def add_step(self, step: Type[AbstractTransformation]) -> None:
        self.steps.append(step)

    def insert_step(self, index: int, step: Type[AbstractTransformation]) -> None:
        self.steps.insert(index, step)

    def remove_step(self, index: int) -> None:
        self.steps.pop(index)

    def clear(self) -> None:
        self.steps.clear()
