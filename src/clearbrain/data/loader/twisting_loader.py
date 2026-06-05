# ================================================================
# 0. Section: IMPORTS
# ================================================================
from pathlib import Path

from ...registration import RegistrationResult
from .file_loader import load_pickle


# ================================================================
# 1. Section: Functions
# ================================================================
def load_twisting_data(path: Path, suffix: str) -> list[RegistrationResult]:
    file_path = path.parent / f"{path.stem}{suffix}.pkl"
    payload = load_pickle(file_path)

    if not isinstance(payload, list) and not all(
        isinstance(item, RegistrationResult) for item in payload
    ):
        raise ValueError(f"Expected a list of RegistrationResult, got {type(payload)}")

    return payload
