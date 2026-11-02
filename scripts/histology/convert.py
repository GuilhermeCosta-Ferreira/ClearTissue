"""Convert a multi-channel histology TIFF into a two-channel HDF5 for register.py.

Output layout (attrs follow cleartissue's DataDownloader convention):
    /background  (Y, X)  channel used to match/register the atlas template
    /signal      (Y, X)  channel whose content we quantify afterwards
    attrs: resolution, unit (JSON, in-plane (y, x)), tissue_type, source, channels

Usage:
    poetry run python scripts/histology/convert.py "data/victor/<file>.tif"
"""

# ================================================================
# 0. Section: IMPORTS
# ================================================================
import h5py
import json
import tifffile
import argparse

from pathlib import Path

from cleartissue.domain_model.data import TissueType


# ================================================================
# 1. Section: CLI
# ================================================================
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert a histology TIFF to HDF5.")
    parser.add_argument("tiff", type=Path, help="Input ImageJ TIFF (C, Y, X)")
    parser.add_argument("--out", type=Path, help="Output .hdf5 (default: next to tiff)")
    parser.add_argument("--background", default="FITC", help="Background channel label")
    parser.add_argument("--signal", default="CY5", help="Signal channel label")
    return parser.parse_args()


# ================================================================
# 2. Section: MAIN
# ================================================================
def main() -> None:
    args = parse_args()
    out = args.out or args.tiff.with_suffix(".hdf5")

    with tifffile.TiffFile(args.tiff) as tif:
        image = tif.asarray()
        if tif.imagej_metadata is None:
            raise ValueError(f"{args.tiff} has no ImageJ channel labels")
        labels = tif.imagej_metadata["Labels"]
        tags = tif.pages.first.tags
        # TIFF stores pixels per unit as a fraction; invert to get unit per pixel
        res_x = tags["XResolution"].value[1] / tags["XResolution"].value[0]
        res_y = tags["YResolution"].value[1] / tags["YResolution"].value[0]

    with h5py.File(out, "w") as f:
        for name, label in (("background", args.background), ("signal", args.signal)):
            f.create_dataset(
                name,
                data=image[labels.index(label)],
                compression="gzip",
                compression_opts=4,
            )
        f.attrs["resolution"] = json.dumps((res_y, res_x))
        f.attrs["unit"] = json.dumps(("um", "um"))
        f.attrs["tissue_type"] = TissueType.SPINAL_CORD.name
        f.attrs["source"] = args.tiff.name
        f.attrs["channels"] = json.dumps({"background": args.background, "signal": args.signal})

    print(f"{out}: {image.shape[1:]} px at {res_y:.4f} x {res_x:.4f} um/px")


if __name__ == "__main__":
    main()
