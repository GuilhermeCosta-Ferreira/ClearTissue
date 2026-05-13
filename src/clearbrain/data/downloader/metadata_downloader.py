# ================================================================
# 0. Section: IMPORTS
# ================================================================
from pathlib import Path
from ..Metadata import Metadata
from .json_downloader import download_json


# ================================================================
# 1. Section: Functions
# ================================================================
def download_metadata(
    source_filepath: Path, metadata: Metadata, to_update: bool, suffix: str
) -> Path:
    # 1. Load the needed variables
    meta = metadata.dict
    return download_json(meta, source_filepath, to_update, suffix)
