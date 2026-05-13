# ================================================================
# 0. Section: IMPORTS
# ================================================================
import os

import numpy as np
import nibabel as nib

from pathlib import Path

from clearbrain.tissue import ClearVolume, TissueType
from clearbrain.data import TissueLoader, TissueSource

# ================================================================
# 1. Section: INPUTS
# ================================================================
DATA_FOLDER: Path = Path("data")
MOUSE: str = "32B"
TISSUE_TYPE: TissueType = TissueType.SPINAL_COORD

TO_UPDATE: bool = True
VOLUME_MODALITIES: list[str] = [
    "_volume",
    "_untwisted",
]


# ================================================================
# 2. Section: FUNCTIONS
# ================================================================
def save_volume_as_nifty(
    source_file_path: Path,
    tissue_volume: ClearVolume,
    to_update: bool = False,
    data_type: type = np.uint8,
    suffix: str = "_volume",
) -> Path:
    volume_data = tissue_volume.volume.astype(data_type)
    file_name = (
        f"{source_file_path.stem}{suffix}_SF{tissue_volume.sample_factor}.nii.gz"
    )
    file_path = tissue_volume.metadata.file_path.parent / file_name

    # 1.A Handles update edge-cases to avoid unwanted overwrite
    if os.path.exists(file_path) and not to_update:
        raise FileExistsError(
            f"File already exists under {file_path}. If you want to update it"
            "make the variable `to_update` to True"
        )

    # 2. Saves the file
    affine = np.eye(4)
    nii = nib.Nifti1Image(volume_data, affine)
    nib.save(nii, file_path)
    return file_path


# ================================================================
# 3. Section: MAIN
# ================================================================
if __name__ == "__main__":
    source = TissueSource(mouse=MOUSE, tissue_type=TISSUE_TYPE, base_path=DATA_FOLDER)
    loader = TissueLoader(source)

    for suffix in VOLUME_MODALITIES:
        tissue = loader.load_volume(suffix=suffix)

        p = save_volume_as_nifty(
            source.source_filepath, tissue, to_update=TO_UPDATE, suffix="_untwisted"
        )
        print(f"Downloaded at {p}")
