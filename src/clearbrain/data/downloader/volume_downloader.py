# ================================================================
# 0. Section: IMPORTS
# ================================================================
import os

import numpy as np

from pathlib import Path
from ...tissue import ClearVolume


# ================================================================
# 1. Section: Functions
# ================================================================
def download_volume(
    source_filepath: Path, volume: ClearVolume, to_update: bool, suffix: str
) -> Path:
    # 1. Load the needed variables
    volume_data = volume.volume
    sample_factor = volume.sample_factor
    file_path = (
        source_filepath.parent / f"{source_filepath.stem}{suffix}_SF{sample_factor}.npy"
    )

    # 1.A Handles update edge-cases to avoid unwanted overwrite
    if os.path.exists(file_path) and not to_update:
        raise FileExistsError(
            f"File already exists under {file_path}. If you want to update it"
            "make the variable `to_update` to True"
        )

    # 2. Saves the file
    np.save(file_path, volume_data)
    return file_path
