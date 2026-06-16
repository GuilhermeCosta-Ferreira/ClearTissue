# ================================================================
# 0. Section: IMPORTS
# ================================================================
from pathlib import Path

from ..load import load_json



# ================================================================
# 1. Section: Functions
# ================================================================
def get_json_cell_data(raw_path: Path, cells_filename: str) -> list:
    # 1.Finds all json on the raw folder
    raw_files = list(raw_path.glob("*.json"))

    # 2. If non are present raise
    if not raw_files:
        raise FileNotFoundError(f"No JSON files found in {raw_path}")

    # 3. If there is only one:
    if len(raw_files) == 1:
        # 3.1 if it has the correct name just load
        if raw_files[0].name == cells_filename:
            return load_json(raw_files[0], list)  # type: ignore

        # 3.2 if does not load it anyways, present a warning and rename it (controllable)
        else:
            print(f"Warning: loaded {raw_files[0]} instead of {cells_filename}")
            print(f"Renamed {raw_files[0]} to {raw_path / cells_filename}")
            raw_files[0].rename(raw_path / cells_filename)
            return load_json(raw_files[0], list)  # type: ignore

    # 4. If there are multiple
    if cells_filename in [f.name for f in raw_files]:
        # 4.1 if the correct name is present, load it
        return load_json(raw_path / cells_filename, list)  # type: ignore

    # 5. if not, raise an error
    raise FileNotFoundError(f"Expected {cells_filename} but found {[f.name for f in raw_files]}")
