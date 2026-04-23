# ================================================================
# 0. Section: IMPORTS
# ================================================================
from pathlib import Path
from ..data.loader import load_json



# ================================================================
# 1. Section: Functions
# ================================================================
def load_metadata(path: Path) -> dict:
    payload = load_json(path)

    if not isinstance(payload, dict):
        raise TypeError(f"The loaded file did not contain a dict ({type(payload)})")

    return payload
