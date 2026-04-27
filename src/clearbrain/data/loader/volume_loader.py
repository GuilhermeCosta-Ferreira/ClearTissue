# ================================================================
# 0. Section: IMPORTS
# ================================================================
from pathlib import Path

from ...tissue import ClearVolume
from .file_loader import load_npy
from .metadata_loader import load_metadata



# ================================================================
# 2. Section: Volume
# ================================================================
def load_volume(source_filepath: Path) -> ClearVolume:
    volume_path = source_filepath.parent / f"{source_filepath.stem}_volume.json"

    data = load_volume_data(volume_path)
    metadata = load_metadata(source_filepath)

    return ClearVolume(data, metadata)


# ──────────────────────────────────────────────────────
# 2.1 Subsection: Load the individual parts
# ──────────────────────────────────────────────────────
def load_volume_data(path: Path):
    volume = load_npy(path)

    if volume.ndim != 3:
        raise ValueError(f"The loaded volume need to be of shape (L, W, H) ({volume.shape})")

    return volume
