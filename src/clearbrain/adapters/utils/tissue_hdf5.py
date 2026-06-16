# ================================================================
# 0. Section: IMPORTS
# ================================================================
from pathlib import Path



# ================================================================
# 1. Section: Functions
# ================================================================
def get_raw_tissue_path(raw_path: Path, tissue_filename: str) -> Path:
    # 1.Finds all hdf5 on the raw folder
    raw_files = list(raw_path.glob("*.hdf5"))

    # 2. Raises an error if no hdf5 files are found
    if not raw_files:
        raise FileNotFoundError(f"No hdf5 files found in {raw_path}")

    # 3. If there is only one:
    if len(raw_files) == 1:
        # 3.1 if it has the correct name just load
        if raw_files[0].name == tissue_filename:
            return raw_files[0]

        # 3.2 if does not load it anyways, present a warning and rename it (controllable)
        else:
            print(f"Warning: loaded {raw_files[0]} instead of {tissue_filename}")
            print(f"Renamed {raw_files[0]} to {raw_path / tissue_filename}")
            raw_files[0].rename(raw_path / tissue_filename)
            return raw_files[0]

    # 4. If there are multiple
    if tissue_filename in [f.name for f in raw_files]:
        # 4.1 if the correct name is present, load it
        return raw_path / tissue_filename

    # 5. if not, raise an error
    raise FileNotFoundError(f"Expected {tissue_filename} but found {[f.name for f in raw_files]}")
