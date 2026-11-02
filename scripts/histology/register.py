"""Register the best-matching spinal cord atlas slices onto a histology section.

1. Downsample the convert.py background to the atlas in-plane resolution.
2. Score every atlas template slice (in all 8 flips/rotations) by normalised
   cross-correlation against the background, both cropped to their bounding box.
3. For the top-k distinct templates: outline affine + B-spline register the template to the tissue,
   resample the atlas annotation through both transforms onto the native grid.

Writes into <out>/<timestamp>/ one <stem>_atlas_<idx>.hdf5 per slice (/annotation on the tissue grid,
same attrs as the input) plus debug plots in debug/ next to them.

Usage:
    poetry run python scripts/histology/register.py "data/victor/<file>.hdf5"
"""

# ================================================================
# 0. Section: IMPORTS
# ================================================================
import h5py
import json
import argparse

import numpy as np
import SimpleITK as sitk
import matplotlib.pyplot as plt

from tqdm import tqdm
from pathlib import Path
from brainglobe_atlasapi import BrainGlobeAtlas
from brainglobe_atlasapi.atlas_name import AtlasName
from datetime import datetime
from dataclasses import replace
from numpy.typing import NDArray
from skimage.filters import gaussian, threshold_otsu
from skimage.segmentation import find_boundaries
from skimage.transform import rescale, resize

from cleartissue.domain_model.data import Atlas
from cleartissue.domain_model.analysis.spinal_level import _level_bounds
from cleartissue.registration import Registrator, RegistratorResampler
from cleartissue.registration.configs import (
    TEMPLATE_AFFINE_REGISTRATION,
    TEMPLATE_WARP_REGISTRATION,
)
from cleartissue.registration.strategies import AffineRegistration, BSplineRegistration


# ================================================================
# 1. Section: Constants
# ================================================================
MATCH_SIZE = (128, 128)
ORIENTATIONS = [(k, flip) for flip in (False, True) for k in range(4)]


# ================================================================
# 2. Section: CLI
# ================================================================
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Register atlas slices to a section.")
    parser.add_argument("hdf5", type=Path, help="Output of convert.py")
    parser.add_argument("--out", type=Path, help="Output folder (default: next to input)")
    parser.add_argument("--atlas", default="allen_cord_20um", help="BrainGlobe atlas")
    parser.add_argument("--top-k", type=int, default=3, help="Atlas slices to register")
    parser.add_argument("--segments", nargs="+", help="Only consider these, e.g. C5 C6 T1")
    parser.add_argument("--grid", type=int, default=2, help="The amount of points per grid")
    parser.add_argument("--warp-method", type=str, default="MI", help="The warp method")
    return parser.parse_args()


# ================================================================
# 3. Section: MAIN
# ================================================================
def main() -> None:
    args = parse_args()
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out = (args.out or args.hdf5.parent / f"{args.hdf5.stem}_registered") / stamp
    (out / "debug").mkdir(parents=True, exist_ok=True)

    with h5py.File(args.hdf5, "r") as f:
        background, signal = np.asarray(f["background"]), np.asarray(f["signal"])
        attrs = dict(f.attrs)
    atlas = Atlas.from_name(args.atlas)

    # asr atlas: axis 0 is rostro-caudal, so slices are (DV, LR) at res[1:]
    scale = np.array(json.loads(str(attrs["resolution"]))) / np.array(atlas.resolution[1:])
    tissue = normalise(rescale(background.astype(np.float32), scale, anti_aliasing=True))

    allowed = segment_mask(args.atlas, args.segments, len(atlas.tissue_template))
    scores, orients = score_slices(tissue, atlas.tissue_template, allowed)
    picks = top_k(scores, atlas.tissue_template, args.top_k)
    plot_scores(scores, orients, picks, out / "debug" / "correlation.png")

    affine = Registrator(AffineRegistration(), RegistratorResampler(), TEMPLATE_AFFINE_REGISTRATION)
    warp_config = replace(
        TEMPLATE_WARP_REGISTRATION,
        metric=replace(TEMPLATE_WARP_REGISTRATION.metric, name=args.warp_method),
        optimizer=replace(TEMPLATE_WARP_REGISTRATION.optimizer, grid_size=args.grid),
    )
    warp = Registrator(BSplineRegistration(), RegistratorResampler(), warp_config)
    for idx in tqdm(picks, desc="registering"):
        k, flip = ORIENTATIONS[orients[idx]]
        template = normalise(orient(atlas.tissue_template[idx], k, flip))
        labels = np.ascontiguousarray(orient(atlas.data[idx], k, flip))
        # Sections shrink; match areas up front since the affine converges slowly
        s = np.sqrt(tissue_mask(tissue).sum() / tissue_mask(template).sum())
        template = rescale(template, s, anti_aliasing=True)

        # Template and FITC contrast differ, so align outlines before intensities
        a = affine.register(gaussian(tissue_mask(tissue), 2), gaussian(tissue_mask(template), 2))
        a_template = np.asarray(affine.apply(tissue, template, a.transform, as_array=True))
        w = warp.register(tissue, a_template)

        # Resample the raw atlas labels once, straight onto the native tissue grid
        native = on_grid(np.zeros(background.shape, np.uint8), scale)
        moving = on_grid(labels, np.array([s, s]))
        both = sitk.CompositeTransform([a.transform, w.transform])
        annotation = np.asarray(warp.apply(native, moving, both, as_array=True))
        # Exact resize: integer striding leaves a few % extra and skews the overlay
        w_labels = np.asarray(resize(annotation, tissue.shape, order=0,
                                     preserve_range=True, anti_aliasing=False))
        save(out / f"{args.hdf5.stem}_atlas_{idx:04d}.hdf5", annotation, attrs,
             idx, scores[idx], k, flip, args.atlas)
        plot_registration(tissue, template, a_template, w.registered_image,
                          w_labels, rescale(signal.astype(np.float32), scale),
                          f"slice {idx}  ncc={scores[idx]:.3f}  rot90={k} flip={flip}",
                          out / "debug" / f"slice_{idx:04d}.png")
        tqdm.write(f"slice {idx}: ncc={scores[idx]:.3f} affine={a.final_metric:.3f} "
              f"warp={w.final_metric:.3f}")


# ──────────────────────────────────────────────────────
# 3.1 Subsection: Helper Functions
# ──────────────────────────────────────────────────────
def on_grid(image: NDArray, spacing: NDArray) -> sitk.Image:
    """Wrap image in the registration's frame: 10 um tissue pixels, centres at 0."""
    out = sitk.GetImageFromArray(image)
    out.SetSpacing([float(v) for v in spacing[::-1]])
    out.SetOrigin([float(v) for v in 0.5 * spacing[::-1] - 0.5])
    return out


def normalise(image: NDArray) -> NDArray:
    lo, hi = np.percentile(image, (1, 99.5))
    return np.clip((image - lo) / (hi - lo + 1e-9), 0, 1).astype(np.float32)


def orient(image: NDArray, k: int, flip: bool) -> NDArray:
    return np.rot90(np.fliplr(image) if flip else image, k)


def tissue_mask(image: NDArray) -> NDArray:
    smooth = gaussian(image, 2)
    return (smooth > threshold_otsu(smooth)).astype(np.float32)


def crop_to_tissue(image: NDArray) -> NDArray:
    """Crop to the foreground bbox and resize, removing scale and position."""
    mask = tissue_mask(image)
    rows, cols = np.flatnonzero(mask.any(1)), np.flatnonzero(mask.any(0))
    crop = image[rows[0]:rows[-1] + 1, cols[0]:cols[-1] + 1]
    crop = np.asarray(resize(crop, MATCH_SIZE, anti_aliasing=True))
    return (crop - crop.mean()) / (crop.std() + 1e-9)


def segment_mask(atlas_name: AtlasName, segments: list[str] | None, n_slices: int) -> NDArray:
    """Which atlas slices belong to the requested segments (all if None)."""
    if not segments:
        return np.ones(n_slices, bool)
    bounds = _level_bounds(BrainGlobeAtlas(atlas_name).metadata["atlas_segments"], n_slices)
    unknown = set(segments) - {name for name, _, _ in bounds}
    if unknown:
        raise ValueError(f"Unknown segments {sorted(unknown)}; atlas has {[b[0] for b in bounds]}")
    mask = np.zeros(n_slices, bool)
    for name, start, end in bounds:
        mask[start:end] |= name in segments
    return mask


def score_slices(tissue: NDArray, template: NDArray, allowed: NDArray) -> tuple[NDArray, NDArray]:
    """Best NCC per atlas slice over all orientations, and which orientation won."""
    target = crop_to_tissue(tissue)
    scores = np.full((len(template), len(ORIENTATIONS)), -1.0)
    for i, slice_ in enumerate(tqdm(template, desc="scoring atlas slices")):
        if not allowed[i] or slice_.max() == 0:
            continue
        for j, (k, flip) in enumerate(ORIENTATIONS):
            scores[i, j] = np.mean(target * crop_to_tissue(orient(slice_, k, flip)))
    return scores.max(1), scores.argmax(1)


def top_k(scores: NDArray, template: NDArray, k: int) -> list[int]:
    """Best k slices, skipping repeats: the template is constant within segments."""
    picks: list[int] = []
    for idx in np.argsort(scores)[::-1]:
        if all(not np.array_equal(template[idx], template[p]) for p in picks):
            picks.append(int(idx))
        if len(picks) == k:
            break
    return picks


def save(path: Path, annotation: NDArray, attrs: dict, idx: int, score: float,
         k: int, flip: bool, atlas_name: str) -> None:
    with h5py.File(path, "w") as f:
        f.create_dataset("annotation", data=annotation, compression="gzip", compression_opts=4)
        f.attrs.update(attrs)
        f.attrs["atlas"] = atlas_name
        f.attrs["atlas_slice"] = idx
        f.attrs["ncc"] = score
        f.attrs["orientation"] = json.dumps({"rot90": k, "fliplr": flip})


def plot_scores(scores: NDArray, orients: NDArray, picks: list[int], path: Path) -> None:
    fig, (ax, bx) = plt.subplots(2, 1, figsize=(12, 6), sharex=True)
    ax.plot(scores, lw=0.8)
    ax.scatter(picks, scores[picks], c="r", zorder=3)
    ax.set_ylabel("best NCC")
    bx.scatter(np.arange(len(orients)), orients, s=2)
    bx.set_yticks(range(len(ORIENTATIONS)), [f"rot{k} {'flip' if f else ''}" for k, f in ORIENTATIONS])
    bx.set_xlabel("atlas slice (rostral → caudal)")
    fig.tight_layout()
    fig.savefig(path, dpi=100)
    plt.close(fig)


def plot_registration(tissue: NDArray, template: NDArray, affine: NDArray, warped: NDArray,
                      labels: NDArray, signal: NDArray, title: str, path: Path) -> None:
    edges = np.ma.masked_where(~find_boundaries(labels.astype(int)), labels)
    panels = [
        ("tissue (background)", tissue, None),
        ("atlas template (oriented)", template, None),
        ("affine: tissue vs template", np.dstack([tissue, normalise(affine), tissue]), None),
        ("warp: tissue vs template", np.dstack([tissue, normalise(warped), tissue]), None),
        ("annotation on background", tissue, edges),
        ("annotation on signal", normalise(signal), edges),
    ]
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    for ax, (name, image, overlay) in zip(axes.flat, panels):
        ax.imshow(image, cmap="gray")
        if overlay is not None:
            ax.imshow(overlay, cmap="tab20", interpolation="nearest")
        ax.set_title(name)
        ax.axis("off")
    fig.suptitle(f"{title}   (magenta = tissue only, green = template only)")
    fig.tight_layout()
    fig.savefig(path, dpi=100)
    plt.close(fig)


if __name__ == "__main__":
    main()
