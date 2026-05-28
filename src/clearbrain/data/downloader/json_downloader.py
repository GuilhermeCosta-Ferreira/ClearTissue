# ================================================================
# 0. Section: IMPORTS
# ================================================================
import os
import json

from pathlib import Path


# ================================================================
# 1. Section: Functions
# ================================================================
def download_json(data: dict, source_filepath: Path, to_update: bool, suffix: str):
    file_path = source_filepath.parent / f"{source_filepath.stem}{suffix}.json"

    # 1.A Handles update edge-cases to avoid unwanted overwrite
    if os.path.exists(file_path) and not to_update:
        raise FileExistsError(
            f"File already exists under {file_path}. If you want to update it"
            "make the variable `to_update` to True"
        )

    # 2. Saves the file
    with file_path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
    return file_path
