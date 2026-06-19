# ================================================================
# 0. Section: IMPORTS
# ================================================================
import json

from pathlib import Path



# ================================================================
# 1. Section: Functions
# ================================================================
def load_json(path: Path, dtype: type = dict) -> dict | list:
    # A. Makes sure there is a file
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")

    with open(str(path), "r", encoding="utf-8") as f:
        payload = json.load(f)

    if type(payload) is not dtype:
        raise TypeError(f"Expected {dtype.__name__} but got {type(payload).__name__}")

    return payload
