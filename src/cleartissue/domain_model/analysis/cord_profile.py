# ================================================================
# 0. Section: IMPORTS
# ================================================================
import warnings

import numpy as np

from typing import Callable, Literal
from numpy.typing import NDArray
from dataclasses import dataclass
from scipy.optimize import curve_fit
from scipy.ndimage import (
    binary_dilation, binary_erosion, gaussian_filter1d, label, median_filter,
)

from ..transformations.utils.size_match import find_best_shift_by_correlation

Direction = Literal["horizontal", "vertical"]
ProfileMode = Literal[
    "horizontal", "vertical", "area", "pca1", "pca2", "pca_ratio",
    "pca1_spread", "pca2_spread", "pca_spread_ratio",
]
# Square kernel: erosion then dilation restores convex corners exactly.
_SQUARE: NDArray = np.ones((3, 3), dtype=bool)
MODES: tuple[ProfileMode, ...] = (
    "horizontal", "vertical", "area", "pca1", "pca2", "pca_ratio",
    "pca1_spread", "pca2_spread", "pca_spread_ratio",
)
ProfileFn = Callable[[NDArray], NDArray]



# ================================================================
# 1. Section: Types
# ================================================================
@dataclass(frozen=True)
class CurveSummary:
    """Per-atlas-slice mean / std / count over a group of aligned curves."""
    mean: NDArray
    std: NDArray
    count: NDArray



@dataclass(frozen=True)
class HeadAlignment:
    """Per-slice atlas matches for the cord's head and the voted shift.

    atlas_slices / scores are (n_head, top_k); column 0 is the best match.
    Atlas slice indices are the midpoint of each run of identical slices.
    """
    shift: int
    tissue_slices: NDArray
    atlas_slices: NDArray
    scores: NDArray
    vote_shifts: NDArray  # candidate shift for each vote bin
    vote: NDArray  # smoothed weighted vote per candidate shift



@dataclass(frozen=True)
class PcaProfiles:
    """Per-slice PCA lengths (voxels) and PC1 direction; see pca_elongation_profiles."""
    pc1: NDArray
    pc2: NDArray
    pc1_spread: NDArray
    pc2_spread: NDArray
    pc1_horizontal: NDArray


@dataclass(frozen=True)
class BumpFit:
    """Gaussian + linear baseline fitted to a per-slice curve (slice units)."""
    centre: float
    sigma: float
    amplitude: float
    smoothed: NDArray  # the median-smoothed curve that was fitted
    model: NDArray  # fitted Gaussian + baseline, per slice

    def excluded(self, n_sigma: float) -> NDArray:
        """Boolean mask of slices within centre ± n_sigma * sigma."""
        x = np.arange(len(self.model))
        return np.abs(x - self.centre) <= n_sigma * self.sigma



# ================================================================
# 2. Section: Functions
# ================================================================
def keep_main_component(volume: NDArray, erosion_iterations: int, restore: bool = True) -> NDArray:
    """Per slice: erode, keep the largest connected group, optionally regrow.

    Erosion breaks thin bridges to debris so the cord separates from it. With
    restore, the kept group is dilated back by the same amount and clipped to
    the original mask, so widths aren't biased down by the erosion. Returns a
    boolean mask; slices that erode to nothing fall back to the raw mask's
    largest group.
    """
    mask = volume > 0
    out = np.zeros_like(mask)

    for sl in range(mask.shape[0]):
        original = mask[sl]
        if not original.any():
            continue
        eroded = (
            binary_erosion(original, _SQUARE, iterations=erosion_iterations)
            if erosion_iterations > 0 else original
        )
        if not eroded.any():
            eroded = original
        kept = _largest_component(eroded)
        if restore and erosion_iterations > 0:
            kept = binary_dilation(kept, _SQUARE, iterations=erosion_iterations) & original
        out[sl] = kept

    return out

def elongation_profile(volume: NDArray, direction: Direction) -> NDArray:
    """Voxel count through each slice's centroid, along one in-plane axis.

    Same measure size matching correlates on; empty slices give 0 instead of
    crashing on an undefined centroid.
    """
    axis = 0 if direction == "horizontal" else 1
    profile = np.zeros(volume.shape[0], dtype=float)

    for sl in range(volume.shape[0]):
        mask = volume[sl] > 0
        coords = np.argwhere(mask)
        if coords.size == 0:
            continue
        center = np.round(coords.mean(axis=0)).astype(int)
        line = mask[center[0], :] if axis == 0 else mask[:, center[1]]
        profile[sl] = np.count_nonzero(line)

    return profile

def area_profile(volume: NDArray) -> NDArray:
    """Non-zero voxel count per slice (cross-sectional area)."""
    return np.count_nonzero(volume.reshape(volume.shape[0], -1) > 0, axis=1).astype(float)

def pca_elongation_profiles(volume: NDArray) -> PcaProfiles:
    """Per-slice mask lengths along its principal axes, plus PC1 direction.

    Two length measures per axis (voxels):
    - extent: max - min of the projected pixel coordinates, +1 so a single
      pixel measures 1. Exact for clean masks, but one stray pixel sets it.
    - spread: 4 * sqrt(eigenvalue). Equals the full axis length of a filled
      ellipse; stray pixels shift it only slightly.
    PC1 is the higher-variance axis. pc1_horizontal is |cos| between PC1 and
    the horizontal (column) axis: 1 = horizontal, 0 = vertical. Empty slices
    give 0 lengths and NaN direction.
    """
    n = volume.shape[0]
    pc1, pc2 = np.zeros(n), np.zeros(n)
    pc1_spread, pc2_spread = np.zeros(n), np.zeros(n)
    pc1_horizontal = np.full(n, np.nan)

    for sl in range(volume.shape[0]):
        coords = np.argwhere(volume[sl] > 0).astype(float)
        if len(coords) == 0:
            continue
        if len(coords) < 3:  # too few points for a stable covariance
            pc1[sl] = pc2[sl] = pc1_spread[sl] = pc2_spread[sl] = 1.0
            continue
        centred = coords - coords.mean(axis=0)
        # Population covariance (ddof=0) so a filled ellipse gives exactly 4·sd.
        eigvals, eigvecs = np.linalg.eigh(np.cov(centred, rowvar=False, ddof=0))
        projected = centred @ eigvecs[:, ::-1]  # eigh is ascending; PC1 first
        extents = np.ptp(projected, axis=0) + 1.0
        pc1[sl], pc2[sl] = extents[0], extents[1]
        spreads = 4.0 * np.sqrt(np.clip(eigvals[::-1], 0.0, None))
        pc1_spread[sl], pc2_spread[sl] = spreads[0], spreads[1]
        pc1_horizontal[sl] = abs(eigvecs[1, -1])  # column component of PC1

    return PcaProfiles(pc1, pc2, pc1_spread, pc2_spread, pc1_horizontal)

def find_atlas_shift(tissue_profile: NDArray, atlas_profile: NDArray) -> int:
    """Shift s such that tissue slice i sits on atlas slice i + s.

    Prefers shifts where the whole tissue fits inside the atlas, and falls
    back to partial overlap when the tissue is longer than the atlas.
    """
    try:
        shift, _ = find_best_shift_by_correlation(tissue_profile, atlas_profile)
    except ValueError:
        min_overlap = max(2, min(len(tissue_profile), len(atlas_profile)) // 2)
        shift, _ = find_best_shift_by_correlation(
            tissue_profile, atlas_profile, min_overlap, require_full_coverage=False
        )
    return shift

def head_alignment(
    tissue_mask: NDArray,
    atlas_volume: NDArray,
    head_fraction: float,
    top_k: int = 3,
    vote_sigma: float = 2.0,
) -> HeadAlignment:
    """Shift found by matching each head slice of the cord to atlas slices.

    Only the first head_fraction of non-empty tissue slices is used, so shape
    changes further down the cord can't pull the alignment. Each slice is
    scored against every unique atlas slice by Dice overlap after centring
    both masks on their centroid. Every top-k match votes for the shift
    (atlas - tissue index), weighted k..1 by rank and by its score; the vote
    histogram is Gaussian-smoothed (vote_sigma slices) and its peak wins.
    """
    tissue_mask = tissue_mask > 0
    atlas_mask = atlas_volume > 0

    filled = np.flatnonzero(tissue_mask.reshape(len(tissue_mask), -1).any(axis=1))
    if len(filled) == 0:
        raise ValueError("Tissue is empty; nothing to align.")
    n_head = max(1, int(round(len(filled) * head_fraction)))
    head_idx = filled[:n_head]

    groups = _unique_slice_runs(atlas_mask)
    reps = np.array([(s + e - 1) // 2 for s, e in groups])

    canvas = (
        max(tissue_mask.shape[1], atlas_mask.shape[1]),
        max(tissue_mask.shape[2], atlas_mask.shape[2]),
    )
    t_flat = _centred_stack(tissue_mask[head_idx], canvas)
    a_flat = _centred_stack(atlas_mask[reps], canvas)

    inter = t_flat @ a_flat.T
    sizes = t_flat.sum(axis=1)[:, None] + a_flat.sum(axis=1)[None, :]
    with np.errstate(divide="ignore", invalid="ignore"):
        dice = np.where(sizes > 0, 2.0 * inter / sizes, 0.0)

    k = min(top_k, dice.shape[1])
    order = np.argsort(-dice, axis=1)[:, :k]
    atlas_slices = reps[order]
    scores = np.take_along_axis(dice, order, axis=1)

    vote_shifts, vote = _vote_shift(
        head_idx, atlas_slices, scores, len(atlas_mask), len(tissue_mask), vote_sigma
    )
    shift = int(vote_shifts[np.argmax(vote)])
    return HeadAlignment(shift, head_idx, atlas_slices, scores, vote_shifts, vote)

def fit_ratio_bump(curve: NDArray, smooth_slices: int = 15) -> BumpFit | None:
    """Fit one Gaussian peak on a linear baseline; None if the fit fails.

    Meant for the PC1/PC2 ratio, where a cord modification shows up as a
    local bump. The curve is median-smoothed first so single-slice outliers
    don't seed the fit; NaN slices (outside the cord) are ignored.
    """
    finite = np.isfinite(curve)
    if finite.sum() < 10:
        return None
    x = np.arange(len(curve), dtype=float)
    filled = np.interp(x, x[finite], curve[finite])
    smoothed = median_filter(filled, size=max(1, smooth_slices), mode="nearest")
    smoothed[~finite] = np.nan

    xs, ys = x[finite], smoothed[finite]
    base = float(np.median(ys))
    peak = int(np.argmax(ys))
    span = float(xs[-1] - xs[0])
    p0 = [max(ys[peak] - base, 1e-3), xs[peak], max(span / 20, 3.0), base, 0.0]
    bounds = ([0.0, xs[0], 3.0, -np.inf, -np.inf], [np.inf, xs[-1], span / 2, np.inf, np.inf])
    try:
        params, _ = curve_fit(_gauss_line, xs, ys, p0=p0, bounds=bounds, maxfev=20000)
    except (RuntimeError, ValueError):
        return None

    amplitude, centre, sigma = params[:3]
    return BumpFit(float(centre), float(sigma), float(amplitude), smoothed, _gauss_line(x, *params))

def place_on_atlas(curve: NDArray, shift: int, atlas_length: int) -> NDArray:
    """Re-index a per-tissue-slice curve onto atlas slices; NaN where uncovered."""
    placed = np.full(atlas_length, np.nan)
    tissue_idx = np.arange(len(curve))
    atlas_idx = tissue_idx + shift
    inside = (atlas_idx >= 0) & (atlas_idx < atlas_length)
    placed[atlas_idx[inside]] = curve[tissue_idx[inside]]
    return placed

def resample_to_grid(curve: NDArray, n_points: int) -> NDArray:
    """Resample an atlas-indexed curve onto n_points spanning atlas start..end.

    Puts animals processed at different resolutions on one shared x axis;
    NaN gaps stay NaN.
    """
    src_x = np.linspace(0.0, 1.0, len(curve))
    dst_x = np.linspace(0.0, 1.0, n_points)
    finite = np.isfinite(curve)
    if not finite.any():
        return np.full(n_points, np.nan)
    # Fill gaps from finite neighbours so edges don't blend towards zero.
    filled = np.interp(src_x, src_x[finite], curve[finite])
    out = np.interp(dst_x, src_x, filled)
    covered = np.interp(dst_x, src_x, np.isfinite(curve).astype(float))
    out[covered < 0.5] = np.nan
    return out

def smooth_curve(curve: NDArray, median_window: int, sigma: float) -> NDArray:
    """Median then NaN-aware Gaussian smoothing; NaN positions stay NaN.

    The median removes single-slice spikes the Gaussian would smear; the
    Gaussian is normalised by smoothed coverage so values near NaN gaps
    aren't pulled towards zero.
    """
    finite = np.isfinite(curve)
    if not finite.any():
        return curve.copy()

    filled = np.where(finite, curve, np.nanmedian(curve))
    if median_window > 1:
        filled = median_filter(filled, size=median_window, mode="nearest")

    if sigma <= 0:
        filled[~finite] = np.nan
        return filled

    weights = finite.astype(float)
    num = gaussian_filter1d(np.where(finite, filled, 0.0), sigma, mode="nearest")
    den = gaussian_filter1d(weights, sigma, mode="nearest")
    with np.errstate(divide="ignore", invalid="ignore"):
        out = num / den
    out[~finite] = np.nan
    return out

def summarise_curves(curves: NDArray) -> CurveSummary:
    """NaN-aware mean/std over rows of an (n_animals, n_slices) array."""
    curves = np.atleast_2d(curves)
    count = np.sum(np.isfinite(curves), axis=0)
    with warnings.catch_warnings():  # all-NaN columns are expected at the ends
        warnings.simplefilter("ignore", RuntimeWarning)
        mean = np.nanmean(curves, axis=0)
        std = np.nanstd(curves, axis=0)
    return CurveSummary(mean=mean, std=std, count=count)


# ──────────────────────────────────────────────────────
# 2.1 Subsection: Helper Functions
# ──────────────────────────────────────────────────────
def _largest_component(mask: NDArray) -> NDArray:
    labels, n = label(mask)  # 4-connectivity: diagonal touches don't merge groups
    if n <= 1:
        return mask
    sizes = np.bincount(labels.ravel())
    sizes[0] = 0
    return labels == int(np.argmax(sizes))

def _unique_slice_runs(mask: NDArray) -> list[tuple[int, int]]:
    """[start, end) runs of consecutive identical slices (atlas repeats them)."""
    runs: list[tuple[int, int]] = []
    start = 0
    for i in range(1, len(mask) + 1):
        if i == len(mask) or not np.array_equal(mask[i], mask[start]):
            if mask[start].any():  # empty atlas slices can't match anything
                runs.append((start, i))
            start = i
    return runs

def _centred_stack(masks: NDArray, canvas: tuple[int, int]) -> NDArray:
    """Flatten each 2D mask after moving its centroid to the canvas centre."""
    out = np.zeros((len(masks), canvas[0] * canvas[1]), dtype=np.float32)
    cy, cx = canvas[0] // 2, canvas[1] // 2
    for i, m in enumerate(masks):
        ys, xs = np.nonzero(m)
        if len(ys) == 0:
            continue
        ys = ys - int(round(ys.mean())) + cy
        xs = xs - int(round(xs.mean())) + cx
        keep = (ys >= 0) & (ys < canvas[0]) & (xs >= 0) & (xs < canvas[1])
        out[i, ys[keep] * canvas[1] + xs[keep]] = 1.0
    return out

def _vote_shift(
    tissue_idx: NDArray, atlas_slices: NDArray, scores: NDArray,
    atlas_len: int, tissue_len: int, sigma: float,
) -> tuple[NDArray, NDArray]:
    k = atlas_slices.shape[1]
    rank_weight = np.arange(k, 0, -1, dtype=float)  # best match counts k, last 1
    shifts = atlas_slices - tissue_idx[:, None]
    weights = scores * rank_weight[None, :]

    offset = tissue_len  # shifts can be negative; histogram needs >= 0 bins
    hist = np.bincount(
        (shifts + offset).ravel(), weights=weights.ravel(),
        minlength=atlas_len + tissue_len + 1,
    ).astype(float)
    if sigma > 0:
        hist = gaussian_filter1d(hist, sigma, mode="constant")
    return np.arange(len(hist)) - offset, hist

def _gauss_line(x: NDArray, a: float, c: float, s: float, b0: float, b1: float) -> NDArray:
    return a * np.exp(-0.5 * ((x - c) / s) ** 2) + b0 + b1 * x
