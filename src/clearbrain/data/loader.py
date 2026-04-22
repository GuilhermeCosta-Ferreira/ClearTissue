# ================================================================
# 0. Section: IMPORTS
# ================================================================
import os
import json

from pathlib import Path



# ================================================================
# 1. Section: Functions
# ================================================================
def load_json(path: Path) -> dict:
    # A. Makes sure there is a file
    if not os.path.exists(path):
        raise FileNotFoundError(f"File not found: {path}")

    with open(str(path), "r", encoding="utf-8") as f:
        payload = json.load(f)

    return payload
