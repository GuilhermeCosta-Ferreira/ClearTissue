# ================================================================
# 0. Section: IMPORTS
# ================================================================
from pathlib import Path
from clearbrain.data import LoadTissue



# ================================================================
# 1. Section: INPUTS
# ================================================================
DATA_FOLDER: Path = Path("data")
MOUSE: str = "32B"
FILE_TARGET: str = "tissue_sc.json"



# ================================================================
# 3. Section: MAIN
# ================================================================
if __name__ == '__main__':
    filepath = DATA_FOLDER / MOUSE / FILE_TARGET
    payload = LoadTissue(filepath).load_tissue()

    print(payload)

    payload.metadata.create_metadata(payload.metadata.file_path)
