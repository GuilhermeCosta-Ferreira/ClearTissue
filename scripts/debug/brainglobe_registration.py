# ================================================================
# 0. Section: IMPORTS
# ================================================================
from pathlib import Path
import subprocess

import numpy as np
import tifffile



# ================================================================
# 1. Config
# ================================================================
SAMPLE_NPY = Path("data/32B/tissue_sc_tissue_untwisted_cleaned_SF1.npy")
STACK_DIR = Path("work/sample_reference_tiff_stack")
OUTPUT_DIR = Path("work/brainreg_sample_reference")

ATLAS_NAME = "allen_cord_20um"

VOXEL_SIZE_UM = (10, 10, 10)
ORIENTATION = "spl"



# ================================================================
# 2. Convert npy volume to TIFF stack
# ================================================================
def npy_to_tiff_stack(npy_path: Path, output_dir: Path) -> None:
    volume = np.load(npy_path)

    if volume.ndim != 3:
        raise ValueError(f"Expected a 3D volume, got shape {volume.shape}")

    output_dir.mkdir(parents=True, exist_ok=True)

    # Normalise only if needed.
    # brainreg registration generally wants an intensity image.
    volume = volume.astype(np.float32)

    v_min = np.nanmin(volume)
    v_max = np.nanmax(volume)

    if v_max > v_min:
        volume = (volume - v_min) / (v_max - v_min)
        volume = (volume * 65535).astype(np.uint16)
    else:
        volume = np.zeros_like(volume, dtype=np.uint16)

    for z in range(volume.shape[0]):
        tifffile.imwrite(output_dir / f"slice_{z:04d}.tiff", volume[z])



# ================================================================
# 3. Run brainreg
# ================================================================
def run_brainreg(
    stack_dir: Path,
    output_dir: Path,
    voxel_size_um: tuple[float, float, float],
    orientation: str,
    atlas_name: str,
) -> None:
    cmd = [
        "brainreg",
        str(stack_dir),
        str(output_dir),
        "-v",
        str(voxel_size_um[0]),
        str(voxel_size_um[1]),
        str(voxel_size_um[2]),
        "--orientation",
        orientation,
        "--atlas",
        atlas_name,

        # Conservative first attempt.
        "--grid-spacing",
        "-20",
        "--bending-energy-weight",
        "0.95",
    ]

    subprocess.run(cmd, check=True)


if __name__ == "__main__":
    npy_to_tiff_stack(SAMPLE_NPY, STACK_DIR)
    run_brainreg(
        stack_dir=STACK_DIR,
        output_dir=OUTPUT_DIR,
        voxel_size_um=VOXEL_SIZE_UM,
        orientation=ORIENTATION,
        atlas_name=ATLAS_NAME,
    )
