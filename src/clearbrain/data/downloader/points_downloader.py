# ================================================================
# 0. Section: IMPORTS
# ================================================================
from pathlib import Path
from ...tissue import ClearTissue
from .json_downloader import download_json


# ================================================================
# 1. Section: Functions
# ================================================================
def download_points(
    source_filepath: Path, tissue: ClearTissue, to_update: bool, suffix: str
) -> Path:
    # 1. Load the needed variables
    points = tissue.points.tolist()
    return download_json(points, source_filepath, to_update, suffix)
