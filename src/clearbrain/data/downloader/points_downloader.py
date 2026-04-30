# ================================================================
# 0. Section: IMPORTS
# ================================================================
import os
import json

from pathlib import Path
from ...tissue import ClearTissue



# ================================================================
# 1. Section: Functions
# ================================================================
def download_points(
    source_filepath: Path,
    tissue: ClearTissue,
    to_update: bool,
    suffix: str
) -> Path:
    # 1. Load the needed variables
    points = tissue.points.tolist()
    file_path = source_filepath.parent / f"{source_filepath.stem}{suffix}.json"

    # 1.A Handles update edge-cases to avoid unwanted overwrite
    if os.path.exists(file_path) and not to_update:
       raise FileExistsError(
           f"File already exists under {file_path}. If you want to update it"
           "make the variable `to_update` to True"
       )

    # 2. Saves the file
    with file_path.open("w", encoding="utf-8") as f:
        json.dump(points, f, indent=2)
    return file_path
