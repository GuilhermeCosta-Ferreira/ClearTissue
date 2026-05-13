# ================================================================
# 0. Section: IMPORTS
# ================================================================
import os

from pathlib import Path
from .file_loader import load_json

from ...tissue import TissueType
from ..Metadata import Metadata


# ================================================================
# 1. Section: Functions
# ================================================================
def load_metadata(source_filepath: Path) -> Metadata:
    metadata_path = source_filepath.parent / f"{source_filepath.stem}_metadata.json"

    if os.path.exists(metadata_path):
        meta = load_meta_dict(metadata_path)

        return Metadata(
            mouse=meta["mouse"],
            tissue_type=TissueType.from_str(meta["tissue_type"]),
            description=meta["description"],
            file_path=metadata_path,
        )

    print("No metadata file found, instantiating a new one")
    tissue_type = metadata_path.stem.rsplit("_", 2)[-2]

    return Metadata(
        mouse=str(metadata_path.parent.name),
        tissue_type=TissueType.from_str(tissue_type),
        file_path=metadata_path,
    )


# ──────────────────────────────────────────────────────
# 1.1 Subsection: Helper Functions
# ──────────────────────────────────────────────────────
def load_meta_dict(path: Path) -> dict:
    payload = load_json(path)

    if not isinstance(payload, dict):
        raise TypeError(f"The loaded file did not contain a dict ({type(payload)})")

    return payload
