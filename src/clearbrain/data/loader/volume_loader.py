# ================================================================
# 0. Section: IMPORTS
# ================================================================
import re

from pathlib import Path

from ...tissue.ClearVolume import ClearVolume
from .file_loader import load_npy
from .metadata_loader import load_metadata



# ================================================================
# 2. Section: Volume
# ================================================================
def load_volume(source_filepath: Path, suffix: str, sample_factor: int) -> ClearVolume:
    volume_path = source_filepath.parent / f"{source_filepath.stem}{suffix}_SF{sample_factor}.npy"

    data = load_volume_data(volume_path)
    metadata = load_metadata(source_filepath)
    sf = load_scale_factor(volume_path)

    return ClearVolume(data, metadata, sf)


# ──────────────────────────────────────────────────────
# 2.1 Subsection: Load the individual parts
# ──────────────────────────────────────────────────────
def load_volume_data(path: Path):
    volume = load_npy(path)

    if volume.ndim != 3:
        raise ValueError(f"The loaded volume need to be of shape (L, W, H) ({volume.shape})")

    return volume

def find_file_path(path: Path) -> Path:
    pattern = re.compile(rf"^{re.escape(path.stem)}\d+\.json$")
    matches = [
        t_path
        for t_path in path.parent.iterdir()
        if t_path.is_file() and pattern.fullmatch(t_path.name)
    ]

    if len(matches) == 0:
        raise FileNotFoundError(
            f"No volume metadata file found matching {path.stem}<number>.json "
            f"in {path.parent}"
        )

    if len(matches) > 1:
        raise RuntimeError(
            "Multiple volume metadata files found:\n"
            + "\n".join(str(path) for path in matches)
        )

    return matches[0]

def load_scale_factor(path: Path) -> int:
    sf = int(str(path.stem).rsplit("SF")[-1])

    return sf
