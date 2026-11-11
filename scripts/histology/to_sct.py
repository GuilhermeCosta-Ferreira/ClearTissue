"""Export a convert.py background as a NIfTI for SCT's gm_wm_mouse_t1 model.

The model expects a 3D MRI-like volume, so the section is downsampled (default
10 um), stacked --slices times along z (isotropic, mm), and inverted in the cord:
FITC grey matter is bright, while T1 grey matter is darker than white matter.

If a register.py output is given, its rot90/flip says how the section sits
relative to the atlas, which sets the anatomical axes in the NIfTI header.

Usage:
    poetry run python scripts/histology/to_sct.py "data/victor/<file>.hdf5" \
        --registered "data/victor/<stem>_registered/<run>/<stem>_atlas_0185.hdf5"
    # then, on the SCT machine:
    sct_deepseg gm_wm_mouse_t1 -i <file>_sct.nii.gz -qc qc
"""

# ================================================================
# 0. Section: IMPORTS
# ================================================================
import h5py
import json
import argparse

import numpy as np

from pathlib import Path
from numpy.typing import NDArray
from scipy.ndimage import binary_fill_holes
from skimage.filters import gaussian, threshold_otsu
from skimage.transform import rescale

from cleartissue.adapters.utils.nii import make_affine, save_nii_gz


# ================================================================
# 1. Section: Constants
# ================================================================
# BrainGlobe asr slice: rows dorsal -> ventral, cols right -> left; in SCT's
# convention dorsal is posterior, so index growth points to anterior / left
ATLAS_SLICE_AXES = ("a", "l")
OPPOSITE = {"a": "p", "p": "a", "l": "r", "r": "l"}


# ================================================================
# 2. Section: CLI
# ================================================================
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export a section as NIfTI for SCT.")
    parser.add_argument("hdf5", type=Path, help="Output of convert.py")
    parser.add_argument("--registered", type=Path, help="register.py output, for orientation")
    parser.add_argument("--res", type=float, default=10, help="Output pixel size (um)")
    parser.add_argument("--slices", type=int, default=32, help="Copies stacked along z")
    parser.add_argument("--no-invert", action="store_true", help="Keep FITC contrast")
    parser.add_argument("--out", type=Path, help="Output .nii.gz (default: next to input)")
    return parser.parse_args()


# ================================================================
# 3. Section: MAIN
# ================================================================
def main() -> None:
    args = parse_args()
    out = args.out or args.hdf5.with_name(f"{args.hdf5.stem}_sct.nii.gz")

    with h5py.File(args.hdf5, "r") as f:
        background = np.asarray(f["background"]).astype(np.float32)
        resolution = json.loads(str(f.attrs["resolution"]))

    image = normalise(rescale(background, np.array(resolution) / args.res, anti_aliasing=True))
    if not args.no_invert:
        smooth = gaussian(image, 2)
        # Fill holes so dark spots inside the cord are inverted, not blacked out
        cord = np.asarray(binary_fill_holes(smooth > threshold_otsu(smooth)), bool)
        image = np.where(cord, 1 - image, 0).astype(np.float32)

    rows, cols = section_axes(args.registered)
    affine = make_affine(
        resolution=(args.res, args.res, args.res),
        unit=("um", "um", "um"),
        orientation=f"{rows}{cols}s",
    )
    # nnU-Net resamples z to its MRI spacing; one thin slice rounds to zero
    volume = np.repeat(image[:, :, None], args.slices, axis=2)
    save_nii_gz(volume, affine, out)
    print(f"{out}: {volume.shape} px at {args.res} um iso, orientation {rows}{cols}s")


# ──────────────────────────────────────────────────────
# 3.1 Subsection: Helper Functions
# ──────────────────────────────────────────────────────
def normalise(image: NDArray) -> NDArray:
    lo, hi = np.percentile(image, (1, 99.5))
    return np.clip((image - lo) / (hi - lo + 1e-9), 0, 1).astype(np.float32)


def section_axes(registered: Path | None) -> tuple[str, str]:
    """Anatomical direction of growing row / col index in the tissue image.

    Replays register.py's orient(): fliplr, then np.rot90 k times, on the axes.
    Without a registration the atlas default is assumed and may be wrong.
    """
    rows, cols = ATLAS_SLICE_AXES
    if registered is None:
        return rows, cols
    with h5py.File(registered, "r") as f:
        orientation = json.loads(str(f.attrs["orientation"]))
    if orientation["fliplr"]:
        cols = OPPOSITE[cols]
    for _ in range(orientation["rot90"]):
        # np.rot90 (counter-clockwise): new rows are the old cols reversed
        rows, cols = OPPOSITE[cols], rows
    return rows, cols


if __name__ == "__main__":
    main()
