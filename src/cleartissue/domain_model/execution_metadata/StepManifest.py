# ================================================================
# 0. Section: IMPORTS
# ================================================================
from typing import Any
from dataclasses import dataclass

from .StepStatus import StepStatus



# ================================================================
# 1. Section: Functions
# ================================================================
@dataclass
class StepManifest:
    pipeline_id: int
    step_id: int
    step_name: str
    signature: str
    status: StepStatus

    source_pipeline_id: int | None = None
    source_step_id: int | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "pipeline_id": self.pipeline_id,
            "step_id": self.step_id,
            "step_name": self.step_name,
            "signature": self.signature,
            "status": self.status.value,
            "source_pipeline_id": self.source_pipeline_id,
            "source_step_id": self.source_step_id,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "StepManifest":
        return cls(
            pipeline_id=int(data["pipeline_id"]),
            step_id=int(data["step_id"]),
            step_name=str(data["step_name"]),
            signature=str(data["signature"]),
            status=StepStatus(data["status"]),
            source_pipeline_id=data.get("source_pipeline_id"),
            source_step_id=data.get("source_step_id"),
        )
