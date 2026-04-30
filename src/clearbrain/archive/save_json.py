# ================================================================
# 0. Section: IMPORTS
# ================================================================
import json

from pathlib import Path



# ================================================================
# 1. Section: Functions
# ================================================================
def save_to_json(data, save_folder: Path, filename: str) -> Path:
    out_path = save_folder / filename
    with open(out_path, "w",  encoding="utf-8") as f:
        json.dump(data, f, indent=4)

    return out_path
