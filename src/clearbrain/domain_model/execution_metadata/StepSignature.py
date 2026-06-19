# ================================================================
# 0. Section: IMPORTS
# ================================================================
import json
import hashlib

from typing import Any
from dataclasses import dataclass



# ================================================================
# 1. Section: Functions
# ================================================================
@dataclass(frozen=True)
class StepSignature:
    step_name: str
    params_hash: str
    input_hash: str
    code_hash: str
    SimpleITK_version: str

    @property
    def value(self) -> str:
        return hash_json({
            "step_name": self.step_name,
            "params_hash": self.params_hash,
            "input_hash": self.input_hash,
            "code_hash": self.code_hash,
            "SimpleITK_version": self.SimpleITK_version,
        })

def hash_json(data: dict[str, Any]) -> str:
    payload = json.dumps(
        data,
        sort_keys=True,
        separators=(",", ":"),
    )

    return hashlib.sha256(payload.encode("utf-8")).hexdigest()
