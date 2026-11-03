"""Export register.py annotations as QuPath-importable GeoJSON.

Each atlas region becomes one annotation (a MultiPolygon, holes kept) named and
coloured from the BrainGlobe structures. Coordinates are pixels of the original
TIFF, so open that same TIFF in QuPath and drag the .geojson onto it.

Usage:
    poetry run python scripts/histology/export.py "data/victor/<stem>_registered/<run>/"*.hdf5
"""

# ================================================================
# 0. Section: IMPORTS
# ================================================================
import h5py
import json
import argparse

import numpy as np

from tqdm import tqdm
from typing import cast
from pathlib import Path
from numpy.typing import NDArray
from scipy.ndimage import binary_fill_holes, find_objects
from skimage.measure import approximate_polygon, find_contours, label
from brainglobe_atlasapi import BrainGlobeAtlas
from brainglobe_atlasapi.atlas_name import AtlasName


# ================================================================
# 1. Section: CLI
# ================================================================
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export annotations to QuPath GeoJSON.")
    parser.add_argument("hdf5", type=Path, nargs="+", help="register.py output file(s)")
    parser.add_argument("--min-area", type=int, default=500, help="Drop pieces/holes below (px)")
    parser.add_argument("--tolerance", type=float, default=1.5, help="Simplify outlines (px)")
    return parser.parse_args()


# ================================================================
# 2. Section: MAIN
# ================================================================
def main() -> None:
    args = parse_args()
    for path in args.hdf5:
        with h5py.File(path, "r") as f:
            annotation = np.asarray(f["annotation"])
            structures = BrainGlobeAtlas(cast(AtlasName, str(f.attrs["atlas"]))).structures

        features = []
        for value in tqdm(np.unique(annotation[annotation > 0]), desc=path.stem):
            polygons = region_polygons(annotation == value, args.min_area, args.tolerance)
            if not polygons:
                continue
            info = structures[int(value)]
            features.append({
                "type": "Feature",
                "geometry": {"type": "MultiPolygon", "coordinates": polygons},
                "properties": {
                    "objectType": "annotation",
                    "name": info["acronym"],
                    "classification": {"name": info["name"], "color": info["rgb_triplet"]},
                    "measurements": {"atlas_id": int(value)},
                },
            })

        out = path.with_suffix(".geojson")
        out.write_text(json.dumps({"type": "FeatureCollection", "features": features}))
        print(f"{out}: {len(features)} regions")


# ──────────────────────────────────────────────────────
# 2.1 Subsection: Helper Functions
# ──────────────────────────────────────────────────────
def region_polygons(mask: NDArray, min_area: int, tolerance: float) -> list:
    """GeoJSON MultiPolygon coordinates: one polygon per piece, holes as inner rings."""
    pieces: NDArray = np.asarray(label(mask))
    polygons = []
    for i, box in enumerate(find_objects(pieces), start=1):
        if box is None:
            continue
        piece: NDArray = pieces[box] == i
        if piece.sum() < min_area:
            continue
        filled: NDArray = np.asarray(binary_fill_holes(piece))
        holes: NDArray = np.asarray(label(filled & ~piece))
        rings = [ring(filled, box, tolerance)]
        rings += [ring(holes == h, box, tolerance) for h in range(1, holes.max() + 1)
                  if (holes == h).sum() >= min_area]
        polygons.append(rings)
    return polygons


def ring(mask: NDArray, box: tuple[slice, ...], tolerance: float) -> list[list[float]]:
    """Closed outline of a single blob, in full-image (x, y) pixel coordinates."""
    # Pad so blobs touching the crop edge still get a closed contour
    contour = max(find_contours(np.pad(mask, 1).astype(float), 0.5), key=len)
    contour = approximate_polygon(contour, tolerance)
    # -1 undoes the pad, +0.5 moves from pixel centres to QuPath's pixel corners
    y = contour[:, 0] - 1 + 0.5 + box[0].start
    x = contour[:, 1] - 1 + 0.5 + box[1].start
    return np.column_stack([x, y]).round(2).tolist()


if __name__ == "__main__":
    main()
