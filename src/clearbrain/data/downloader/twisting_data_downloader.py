# ================================================================
# 0. Section: IMPORTS
# ================================================================
import os
import pickle

from pathlib import Path

from ...registration import RegistrationResult



# ================================================================
# 1. Section: Functions
# ================================================================
def download_twisting_data(
    source_filepath: Path,
    twisting_data: list[RegistrationResult],
    to_update: bool,
    suffix: str
) -> Path:
    # 1. Load the needed variables
    file_path = source_filepath.parent / f"{source_filepath.stem}{suffix}.pkl"

    # 1.A Handles update edge-cases to avoid unwanted overwrite
    if os.path.exists(file_path) and not to_update:
       raise FileExistsError(
           f"File already exists under {file_path}. If you want to update it"
           "make the variable `to_update` to True"
       )

    # 2. Saves the file
    with open(file_path, "wb") as f:
        pickle.dump(twisting_data, f, protocol=pickle.HIGHEST_PROTOCOL)
    return file_path
