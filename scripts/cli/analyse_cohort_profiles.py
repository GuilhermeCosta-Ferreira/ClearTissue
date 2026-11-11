"""Compare cord elongation profiles across conditions, aligned to the atlas.

Each animal's tissue profile is shifted onto the atlas by the same
cross-correlation used in size matching, so cervical/thoracic/lumbar line up
across animals. Every profile mode in MODES is generated in one run; per mode
it writes one plot per condition (animals + mean ± std) and one plot of all
condition means, both raw and normalised (1.0 = same size as the denominator
at that level), plus a CSV of every aligned curve. The denominator is a
reference cord (default) or the atlas; see NORMALISE_BY.

Edit the INPUTS section in place, then run:
    poetry run python scripts/cli/analyse_cohort_profiles.py
"""
# ================================================================
# 0. Section: IMPORTS
# ================================================================
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from pathlib import Path
from datetime import datetime
from numpy.typing import NDArray
from typing import Callable, Literal
from dataclasses import dataclass
from matplotlib.axes import Axes

from cleartissue.domain_model.data import TissueType
from cleartissue.adapters.RawDataLoader import RawDataLoader
from cleartissue.service.ClearTissueProject import ClearTissueProject
from cleartissue.domain_model.analysis import spinal_level_bounds
from cleartissue.domain_model.analysis.cord_profile import (
    CurveSummary,
    BumpFit,
    fit_ratio_bump,
    HeadAlignment,
    head_alignment,
    Direction,
    ProfileMode,
    MODES,
    area_profile,
    elongation_profile,
    keep_main_component,
    pca_elongation_profiles,
    find_atlas_shift,
    place_on_atlas,
    resample_to_grid,
    smooth_curve,
    summarise_curves,
)


# ================================================================
# 1. Section: Types
# ================================================================
@dataclass(frozen=True)
class AnimalRun:
    mouse: str
    pipeline_id: int
    step_id: int


# ================================================================
# 2. Section: Inputs
# ================================================================
TISSUE_TYPE: TissueType = TissueType.SPINAL_CORD
CONDITIONS: dict[str, list[AnimalRun]] = {
    #"EES_2mm": [
    #    AnimalRun(mouse="EESpaddle_2mm_1", pipeline_id=2, step_id=7),
    #    #AnimalRun(mouse="EESpaddle_2mm_2", pipeline_id=2, step_id=7),
    #    AnimalRun(mouse="EESpaddle_2mm_3", pipeline_id=2, step_id=7),
    #],
    #"EES_1_85mm": [
    #    AnimalRun(mouse="EESpaddle_185mm_1", pipeline_id=2, step_id=7),
    #    AnimalRun(mouse="EESpaddle_185mm_2", pipeline_id=2, step_id=7),
    #    AnimalRun(mouse="EESpaddle_185mm_3", pipeline_id=2, step_id=7),
    #],
    "EES_1_7mm": [
        AnimalRun(mouse="EESpaddle_17mm_1", pipeline_id=2, step_id=7),
        AnimalRun(mouse="EESpaddle_17mm_2", pipeline_id=2, step_id=7),
        AnimalRun(mouse="EESpaddle_17mm_3", pipeline_id=2, step_id=7),
    ],
    "Control": [
        AnimalRun(mouse="190R", pipeline_id=2, step_id=7),
        AnimalRun(mouse="191R", pipeline_id=2, step_id=7),
        AnimalRun(mouse="193R", pipeline_id=2, step_id=7),
    ],
}

ALIGN_METHOD: Literal["correlation", "head", "pca-corrected"] = "pca-corrected"
ALIGN_DIRECTION: Direction = "horizontal"
HEAD_FRACTION: float = 0.30
HEAD_TOP_K: int = 3

BUMP_CORRECTION: dict[str, bool] = {  # missing groups default to False
    "EES_2mm": True,
    "EES_1_85mm": True,
    "EES_1_7mm": True,
    "Control": False,
}
REFERENCE_BUMP_CORRECTION: bool = False
BUMP_EXCLUDE_SIGMAS: float = 1.5  # wider leaves too little cord; 2.5σ was unstable
# Required for "pca-corrected" (ignored otherwise): the per-slice curve whose
# Gaussian bump marks the modification, e.g. "pca_ratio" or "pca_spread_ratio".
BUMP_MODE: ProfileMode | None = "pca_ratio"
BUMP_SMOOTH_SLICES: int = 15
# Slices shown per cord in diagnostics/<mouse>_bump_slices.png, in σ from centre.
BUMP_GALLERY_SIGMAS: tuple[float, ...] = (-3, -2, -1.5, -1, -0.5, 0, 0.5, 1, 1.5, 2, 3)

CLEAN_EROSION_ITERATIONS: int = 2
CLEAN_RESTORE: bool = True

N_GRID: int = 500
SHOW_SEGMENTS: bool = True
NORMALISE_BY: Literal["reference", "atlas"] = "atlas"
REFERENCE: AnimalRun = AnimalRun(mouse="193R", pipeline_id=2, step_id=7)

ATLAS_MEDIAN_SLICES: int = 5
ATLAS_SMOOTH_SIGMA: float = 6.0

OUTPUT_ROOT: Path = Path("data/cohort_profiles")
_UNITS: dict[str, str] = {
    "horizontal": "horizontal width (µm)",
    "vertical": "vertical width (µm)",
    "area": "cross-section area (µm²)",
    "pca1": "PC1 (long axis) length (µm)",
    "pca2": "PC2 (short axis) length (µm)",
    "pca_ratio": "PC1 / PC2 ratio (unitless)",
    "pca1_spread": "PC1 spread length, 4·sd (µm)",
    "pca2_spread": "PC2 spread length, 4·sd (µm)",
    "pca_spread_ratio": "PC1 / PC2 spread ratio (unitless)",
}


# ================================================================
# 3. Section: Functions
# ================================================================
@dataclass(frozen=True)
class AlignedAnimal:
    """One animal's tissue and atlas curves per mode, on the shared grid."""
    shift: int
    tissue: dict[str, NDArray]
    atlas: dict[str, NDArray]
    pc1_horizontal: float
    head: HeadAlignment | None
    atlas_len: int
    correlation_shift: int
    bump: BumpFit | None
    bump_slices: list[tuple[int, NDArray]]
    bump_curve_raw: NDArray  # per tissue slice BUMP_MODE curve, before alignment
    removed_fraction: NDArray
    worst_slice: tuple[int, NDArray, NDArray]


def align_animal(run: AnimalRun, correct_bump: bool = False) -> AlignedAnimal:
    """Load one animal and compute every mode's aligned curves.

    correct_bump only matters for ALIGN_METHOD == "pca-corrected"; if the bump
    fit fails the plain correlation shift is used.
    """
    project = ClearTissueProject.load(mouse=run.mouse, tissue_type=TISSUE_TYPE)
    batch = project.io.load_batch(pipeline_id=run.pipeline_id, step=run.step_id)

    raw = np.asarray(batch.tissue.data) > 0
    tissue = raw
    if CLEAN_EROSION_ITERATIONS > 0:
        tissue = keep_main_component(raw, CLEAN_EROSION_ITERATIONS, CLEAN_RESTORE)
    before = raw.reshape(len(raw), -1).sum(axis=1)
    after = tissue.reshape(len(tissue), -1).sum(axis=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        removed = np.where(before > 0, 1 - after / before, np.nan)
    worst = int(np.nanargmax(removed)) if np.isfinite(removed).any() else 0
    atlas = np.asarray(batch.atlas.data)
    tissue_um = float(batch.tissue.resolution[1])
    atlas_um = float(batch.atlas.resolution[1])

    tissue_raw, direction = _all_profiles(tissue, tissue_um)
    atlas_raw, _ = _all_profiles(atlas, atlas_um)

    tissue_align = elongation_profile(tissue, ALIGN_DIRECTION)
    atlas_align = elongation_profile(atlas, ALIGN_DIRECTION)

    correlation_shift = find_atlas_shift(tissue_align, atlas_align)

    head, bump = None, None
    shift = correlation_shift
    if ALIGN_METHOD == "head":
        head = head_alignment(tissue, atlas, HEAD_FRACTION, HEAD_TOP_K)
        shift = head.shift
    elif ALIGN_METHOD == "pca-corrected" and correct_bump:
        bump = fit_ratio_bump(tissue_raw[BUMP_MODE], BUMP_SMOOTH_SLICES)
        if bump is None:
            print(f"  {run.mouse}: bump fit failed, using plain correlation shift")
        else:
            masked = tissue_align.copy()
            masked[bump.excluded(BUMP_EXCLUDE_SIGMAS)] = np.nan
            shift = find_atlas_shift(masked, atlas_align)

    atlas_curves = {}
    for mode in MODES:
        a = smooth_curve(atlas_raw[mode], ATLAS_MEDIAN_SLICES, sigma=0)
        atlas_curves[mode] = smooth_curve(resample_to_grid(a, N_GRID), 1, ATLAS_SMOOTH_SIGMA)

    bump_slices: list[tuple[int, NDArray]] = []
    if bump is not None:
        picks = sorted({
            int(round(bump.centre + k * bump.sigma)) for k in BUMP_GALLERY_SIGMAS
            if 0 <= round(bump.centre + k * bump.sigma) < len(tissue)
        })
        bump_slices = [(i, tissue[i].copy()) for i in picks]

    tissue_curves = {
        mode: resample_to_grid(place_on_atlas(tissue_raw[mode], shift, atlas.shape[0]), N_GRID)
        for mode in MODES
    }

    return AlignedAnimal(
        shift=shift, tissue=tissue_curves, atlas=atlas_curves,
        pc1_horizontal=float(np.nanmean(direction)),
        head=head, atlas_len=atlas.shape[0],
        correlation_shift=correlation_shift, bump=bump,
        bump_slices=bump_slices,
        bump_curve_raw=tissue_raw[BUMP_MODE] if BUMP_MODE else np.array([]),
        removed_fraction=removed, worst_slice=(worst, raw[worst], tissue[worst]),
    )

def normaliser(reference: AlignedAnimal | None) -> Callable[[AlignedAnimal, str], NDArray]:
    """Return f(animal, mode) -> animal curve divided by the chosen denominator."""
    def divide(animal: AlignedAnimal, mode: str) -> NDArray:
        denom = animal.atlas[mode] if reference is None else reference.tissue[mode]
        with np.errstate(divide="ignore", invalid="ignore"):
            return animal.tissue[mode] / denom
    return divide

def pca_axis_table(animals: dict[str, list[tuple[str, "AlignedAnimal"]]]) -> pd.DataFrame:
    """Per-animal mean PC1 direction and which in-plane axis it is closer to."""
    rows = [
        {
            "condition": name, "mouse": mouse,
            "pc1_horizontal": round(a.pc1_horizontal, 3),
            "closer_axis": _closer_axis(a.pc1_horizontal),
        }
        for name, group in animals.items() for mouse, a in group
    ]
    return pd.DataFrame(rows)

def head_match_table(mouse: str, animal: AlignedAnimal) -> pd.DataFrame:
    """The (head slices x top-k) match matrix, one row per tissue slice."""
    head = animal.head
    assert head is not None
    levels = _slice_levels(animal.atlas_len)
    frame = pd.DataFrame({"mouse": mouse, "tissue_slice": head.tissue_slices})
    for r in range(head.atlas_slices.shape[1]):
        frame[f"rank{r + 1}_atlas_slice"] = head.atlas_slices[:, r]
        frame[f"rank{r + 1}_level"] = [levels[i] for i in head.atlas_slices[:, r]]
        frame[f"rank{r + 1}_dice"] = head.scores[:, r].round(4)
        frame[f"rank{r + 1}_shift"] = head.atlas_slices[:, r] - head.tissue_slices
    frame["voted_shift"] = head.shift
    return frame

def plot_head_matches(mouse: str, animal: AlignedAnimal, ax: Axes) -> None:
    """Tissue slice vs matched atlas slice per rank; dashed = voted shift."""
    head = animal.head
    assert head is not None
    for r in range(head.atlas_slices.shape[1] - 1, -1, -1):
        ax.scatter(head.tissue_slices, head.atlas_slices[:, r], s=6,
                   alpha=0.4 if r else 0.9, label=f"rank {r + 1}")
    x = np.array([head.tissue_slices[0], head.tissue_slices[-1]])
    ax.plot(x, x + head.shift, "k--", lw=1, label=f"voted shift {head.shift}")
    ax.set_xlabel("tissue slice (head only)")
    ax.set_ylabel("matched atlas slice")
    ax.set_title(f"{mouse}: head-slice matches")
    ax.legend(fontsize=7)

def save_diagnostics(path: Path, mouse: str, animal: AlignedAnimal) -> None:
    """One sheet per cord: cleanup, head matching, and final alignment."""
    fig, axes = plt.subplots(2, 3, figsize=(16, 8.5))
    (a_clean, a_slice, a_match), (a_dice, a_vote, a_final) = axes

    # 1. Cleanup: how much each slice lost, and the worst slice before/after.
    a_clean.plot(animal.removed_fraction * 100, lw=0.8)
    a_clean.set(title="1. cleanup: % of mask removed per slice",
                xlabel="tissue slice", ylabel="% removed")
    idx, before, after = animal.worst_slice
    overlay = np.zeros(before.shape + (3,))
    overlay[before & ~after] = (0.85, 0.2, 0.2)  # removed = red
    overlay[after] = (0.8, 0.8, 0.8)  # kept = grey
    a_slice.imshow(overlay, interpolation="nearest")
    a_slice.set_title(f"1. worst slice {idx}: kept grey, removed red "
                      f"({np.nan_to_num(animal.removed_fraction[idx]) * 100:.1f}%)")
    a_slice.axis("off")

    # 2. Head matching (only when ALIGN_METHOD == "head").
    head = animal.head
    if head is not None:
        plot_head_matches(mouse, animal, a_match)
        a_match.set_title("2. head matches (dashed = voted shift)")
        for r in range(head.scores.shape[1]):
            a_dice.plot(head.tissue_slices, head.scores[:, r], lw=0.8, label=f"rank {r + 1}")
        a_dice.set(title="2. match quality (Dice) per head slice",
                   xlabel="tissue slice", ylabel="Dice", ylim=(0, 1))
        a_dice.legend(fontsize=7)
        keep = head.vote > head.vote.max() * 1e-3
        a_vote.plot(head.vote_shifts[keep], head.vote[keep])
        a_vote.axvline(head.shift, color="k", ls="--", lw=1)
        a_vote.set(title=f"2. shift vote (peak = {head.shift})",
                   xlabel="shift (atlas slice - tissue slice)", ylabel="weighted votes")
    elif animal.bump is not None:
        _plot_bump_panels(animal, a_match, a_dice, a_vote)
    else:
        why = ("pca-corrected: correction off for this group\n(plain correlation)"
               if ALIGN_METHOD == "pca-corrected" else "correlation alignment")
        for ax in (a_match, a_dice, a_vote):
            ax.text(0.5, 0.5, f"{why}\n(shift {animal.shift})",
                    ha="center", va="center", transform=ax.transAxes)
            ax.axis("off")

    # 3. Result: cord profile shifted onto the atlas profile.
    x = _grid_x()
    a_final.plot(x, animal.atlas["horizontal"], color="grey", label="atlas")
    a_final.plot(x, animal.tissue["horizontal"], label="cord (aligned)")
    if animal.bump is not None:  # the stretch the correlation ignored
        lo, hi = _bump_span(animal.bump)
        a_final.axvspan((lo + animal.shift) / animal.atlas_len,
                        (hi + animal.shift) / animal.atlas_len,
                        color="tab:red", alpha=0.12, label="bump ignored")
    if head is not None:  # highlight the part of the cord the shift came from
        start = (head.tissue_slices[0] + head.shift) / animal.atlas_len
        end = (head.tissue_slices[-1] + head.shift) / animal.atlas_len
        a_final.axvspan(start, end, color="tab:orange", alpha=0.12, label="head used")
    a_final.set(title=f"3. aligned horizontal width (shift {animal.shift})",
                xlabel="position along atlas", ylabel="width (µm)", xlim=(0, 1))
    a_final.legend(fontsize=7)

    fig.suptitle(mouse)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)

def save_bump_gallery(path: Path, mouse: str, animal: AlignedAnimal) -> None:
    """Cross-sections before, inside and after the ignored bump, same zoom."""
    bump = animal.bump
    assert bump is not None and animal.bump_slices
    excluded = bump.excluded(BUMP_EXCLUDE_SIGMAS)

    union = np.any([m for _, m in animal.bump_slices], axis=0)
    ys, xs = np.nonzero(union)
    pad = 3
    crop = (slice(max(ys.min() - pad, 0), ys.max() + pad + 1),
            slice(max(xs.min() - pad, 0), xs.max() + pad + 1))

    n = len(animal.bump_slices)
    cols = min(n, 6)
    rows = int(np.ceil(n / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(2.6 * cols, 2.9 * rows + 0.6), squeeze=False)
    for ax in axes.ravel():
        ax.axis("off")
    for ax, (idx, mask) in zip(axes.ravel(), animal.bump_slices):
        inside = bool(excluded[idx])
        ax.imshow(mask[crop], cmap="Reds" if inside else "Greys", vmin=0, vmax=1.4,
                  interpolation="nearest")
        z = (idx - bump.centre) / bump.sigma
        ax.set_title(f"slice {idx} ({z:+.1f}σ)\n{BUMP_MODE} {animal.bump_curve_raw[idx]:.2f}"
                     f"  fit {bump.model[idx]:.2f}", fontsize=8,
                     color="tab:red" if inside else "black")
    fig.suptitle(f"{mouse}: slices around the fitted bump (red = ignored by alignment, "
                 f"±{BUMP_EXCLUDE_SIGMAS:g}σ; centre {bump.centre:.0f}, σ {bump.sigma:.0f})",
                 fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)

def plot_condition(
    name: str, curves: NDArray, labels: list[str], summary: CurveSummary,
    ax: Axes, ylabel: str,
) -> None:
    x = _grid_x()
    for curve, label in zip(curves, labels):
        ax.plot(x, curve, lw=0.8, alpha=0.5, label=label)
    _plot_mean_std(ax, x, summary, color="black", label=f"mean (n={len(curves)})")
    ax.set_title(name)
    _style_axis(ax, ylabel)

def plot_means(summaries: dict[str, CurveSummary], ax: Axes, ylabel: str) -> None:
    x = _grid_x()
    for name, summary in summaries.items():
        _plot_mean_std(ax, x, summary, color=None, label=name)
    ax.set_title("Condition means ± std")
    _style_axis(ax, ylabel)


# ──────────────────────────────────────────────────────
# 3.1 Subsection: Helper Functions
# ──────────────────────────────────────────────────────
def _bump_span(bump: BumpFit) -> tuple[int, int]:
    idx = np.flatnonzero(bump.excluded(BUMP_EXCLUDE_SIGMAS))
    return int(idx[0]), int(idx[-1])

def _plot_bump_panels(animal: AlignedAnimal, a_fit: Axes, a_grid: Axes, a_text: Axes) -> None:
    bump = animal.bump
    assert bump is not None
    lo, hi = _bump_span(bump)

    a_fit.plot(bump.smoothed, lw=0.8, label=f"{BUMP_MODE} (median-smoothed)")
    a_fit.plot(bump.model, "k", lw=1.2, label="Gaussian + baseline fit")
    a_fit.axvspan(lo, hi, color="tab:red", alpha=0.12, label=f"ignored (±{BUMP_EXCLUDE_SIGMAS:g}σ)")
    a_fit.set(title=f"2. bump fit: centre {bump.centre:.0f}, σ {bump.sigma:.0f}, "
                    f"amp {bump.amplitude:.2f}", xlabel="tissue slice", ylabel=BUMP_MODE)
    a_fit.legend(fontsize=7)

    x = _grid_x()
    a_grid.plot(x, animal.tissue[BUMP_MODE], lw=0.8)
    a_grid.axvspan((lo + animal.shift) / animal.atlas_len, (hi + animal.shift) / animal.atlas_len,
                   color="tab:red", alpha=0.12)
    a_grid.set(title=f"2. {BUMP_MODE} after corrected alignment",
               xlabel="position along atlas", ylabel=BUMP_MODE, xlim=(0, 1))

    moved = animal.shift - animal.correlation_shift
    a_text.text(0.5, 0.5,
                f"plain correlation shift: {animal.correlation_shift}\n"
                f"corrected shift: {animal.shift}\n"
                f"moved by {moved:+d} slices ({moved / animal.atlas_len:+.1%} of atlas)\n"
                f"slices ignored: {hi - lo + 1}",
                ha="center", va="center", transform=a_text.transAxes, fontsize=11)
    a_text.axis("off")

def _slice_levels(atlas_len: int) -> list[str]:
    names = [""] * atlas_len
    for name, start, end in spinal_level_bounds(_SEGMENTS, atlas_len) if _SEGMENTS else []:
        for i in range(start, min(end, atlas_len)):
            names[i] = name
    return names

def _axis_note(pc1_horizontal: float) -> str:
    return f" | PC1 closer to {_closer_axis(pc1_horizontal)} (|cos|={pc1_horizontal:.2f})"

def _closer_axis(pc1_horizontal: float) -> str:
    # |cos| > cos(45°) means PC1 is within 45° of the horizontal axis.
    return "horizontal" if pc1_horizontal > np.sqrt(0.5) else "vertical"

def _all_profiles(volume: NDArray, voxel_um: float) -> tuple[dict[str, NDArray], NDArray]:
    """Every mode's raw per-slice curve (µm / µm²) and per-slice PC1 direction."""
    pca = pca_elongation_profiles(volume)
    pc1, pc2, direction = pca.pc1, pca.pc2, pca.pc1_horizontal
    curves = {
        "horizontal": elongation_profile(volume, "horizontal") * voxel_um,
        "vertical": elongation_profile(volume, "vertical") * voxel_um,
        "area": area_profile(volume) * voxel_um ** 2,
        "pca1": pc1 * voxel_um,
        "pca2": pc2 * voxel_um,
        # Unitless, so voxel size cancels; 1.0 = round slice, higher = flatter.
        "pca_ratio": np.divide(pc1, pc2, out=np.zeros_like(pc1), where=pc2 > 0),
        "pca1_spread": pca.pc1_spread * voxel_um,
        "pca2_spread": pca.pc2_spread * voxel_um,
        "pca_spread_ratio": np.divide(
            pca.pc1_spread, pca.pc2_spread,
            out=np.zeros_like(pc1), where=pca.pc2_spread > 0,
        ),
    }
    for curve in curves.values():
        curve[curve == 0] = np.nan  # empty slices are outside the cord, not width 0
    return curves, direction

def _save_figure(path: Path, draw, unity_line: bool = False) -> None:
    fig, ax = plt.subplots(figsize=(10, 4))
    if unity_line:  # 1.0 = same size as the denominator; above = bigger
        ax.axhline(1.0, color="dimgrey", lw=1, ls="--", zorder=0)
    draw(ax)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)

def _grid_x() -> NDArray:
    return np.linspace(0.0, 1.0, N_GRID)

def _plot_mean_std(
    ax: Axes, x: NDArray, summary: CurveSummary, color: str | None, label: str
) -> None:
    (line,) = ax.plot(x, summary.mean, lw=2, color=color, label=label)
    ax.fill_between(
        x, summary.mean - summary.std, summary.mean + summary.std,
        color=line.get_color(), alpha=0.2, lw=0,
    )

def _style_axis(ax: Axes, ylabel: str) -> None:
    ax.set_xlabel("position along atlas (rostral → caudal)")
    ax.set_ylabel(ylabel)
    ax.set_xlim(0, 1)
    if SHOW_SEGMENTS and _SEGMENTS:
        _draw_segments(ax)
    ax.legend(fontsize=7, loc="upper right")

def _draw_segments(ax: Axes) -> None:
    bounds = spinal_level_bounds(_SEGMENTS, N_GRID)
    ticks, names = [], []
    for i, (name, start, end) in enumerate(bounds):
        if i % 2 == 0:
            ax.axvspan(start / N_GRID, end / N_GRID, color="grey", alpha=0.07, lw=0)
        ticks.append((start + end) / 2 / N_GRID)
        names.append(name)
    top = ax.secondary_xaxis("top")
    top.set_xticks(ticks, names, fontsize=6, rotation=90)

def _load_segments() -> list[dict]:
    first = next(run for runs in CONDITIONS.values() for run in runs)
    project = ClearTissueProject.load(mouse=first.mouse, tissue_type=TISSUE_TYPE)
    return RawDataLoader(project.source).load_atlas_segments()


_SEGMENTS: list[dict] = []


# ================================================================
# 4. Section: Main
# ================================================================
def main() -> None:
    global _SEGMENTS
    if ALIGN_METHOD == "pca-corrected" and BUMP_MODE not in MODES:
        raise ValueError(
            f'ALIGN_METHOD "pca-corrected" needs BUMP_MODE set to one of {MODES}, '
            f"got {BUMP_MODE!r}"
        )
    _SEGMENTS = _load_segments()  # also names spinal levels in head matches

    reference: AlignedAnimal | None = None
    if NORMALISE_BY == "reference":
        reference = align_animal(REFERENCE, REFERENCE_BUMP_CORRECTION)
        norm_tag = f"ref-{REFERENCE.mouse}-p{REFERENCE.pipeline_id}-s{REFERENCE.step_id}"
        norm_label = f"/ {REFERENCE.mouse}"
        print(f"[reference] {REFERENCE.mouse}: shift {_shift_note(reference)}")
    else:
        norm_tag, norm_label = "atlas", "/ atlas"

    stamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    clean_tag = (
        f"_clean-e{CLEAN_EROSION_ITERATIONS}{'' if CLEAN_RESTORE else '-norestore'}"
        if CLEAN_EROSION_ITERATIONS > 0 else ""
    )
    align_tag = {
        "head": f"head{round(HEAD_FRACTION * 100)}",
        "pca-corrected": f"pcacorr-{BUMP_MODE}-{BUMP_EXCLUDE_SIGMAS:g}sd",
        "correlation": "corr",
    }[ALIGN_METHOD]
    out_dir = OUTPUT_ROOT / f"{stamp}_align-{align_tag}_norm-{norm_tag}{clean_tag}"
    out_dir.mkdir(parents=True, exist_ok=False)

    animals: dict[str, list[tuple[str, AlignedAnimal]]] = {}
    for name, runs in CONDITIONS.items():
        animals[name] = []
        for run in runs:
            aligned = align_animal(run, BUMP_CORRECTION.get(name, False))
            animals[name].append((run.mouse, aligned))
            print(f"[{name}] {run.mouse}: shift {_shift_note(aligned)}")

    write_section(out_dir, animals, reference, norm_tag, norm_label)
    print(f"Wrote {len(MODES)} modes to {out_dir}/")


def write_section(
    out_dir: Path,
    animals: dict[str, list[tuple[str, AlignedAnimal]]],
    reference: AlignedAnimal | None,
    norm_tag: str,
    norm_label: str,
) -> None:
    """All plots, CSVs and diagnostics for one alignment of the cohort."""
    out_dir.mkdir(parents=True, exist_ok=True)
    normalise = normaliser(reference)

    diag_dir = out_dir / "diagnostics"
    diag_dir.mkdir()
    everyone = [(m, a) for g in animals.values() for m, a in g]
    if reference is not None:
        everyone.append((f"reference_{REFERENCE.mouse}", reference))
    for mouse, a in everyone:
        save_diagnostics(diag_dir / f"{mouse}.png", mouse, a)
        if a.bump is not None and a.bump_slices:
            save_bump_gallery(diag_dir / f"{mouse}_bump_slices.png", mouse, a)
        if a.head is not None:
            head_match_table(mouse, a).to_csv(diag_dir / f"{mouse}_head_matches.csv", index=False)

    pd.DataFrame([
        {
            "condition": name, "mouse": mouse,
            "correlation_shift": a.correlation_shift, "shift": a.shift,
            "bump_corrected": a.bump is not None,
            "bump_centre": a.bump.centre if a.bump else np.nan,
            "bump_sigma": a.bump.sigma if a.bump else np.nan,
            "bump_amplitude": a.bump.amplitude if a.bump else np.nan,
        }
        for name, group in animals.items() for mouse, a in group
    ]).to_csv(out_dir / "alignment.csv", index=False)

    pca_axis = pca_axis_table(animals)
    pca_axis.to_csv(out_dir / "pca1_axis.csv", index=False)
    group_axis = pca_axis.groupby("condition", sort=False)["pc1_horizontal"].mean()

    rows: list[pd.DataFrame] = []
    for mode in MODES:
        mode_dir = out_dir / mode
        mode_dir.mkdir()
        variants = {
            "": (_UNITS[mode], lambda a: a.tissue[mode]),
            "normalised_": (f"{_UNITS[mode].split(' (')[0]} {norm_label}", lambda a: normalise(a, mode)),
        }

        for prefix, (ylabel, pick) in variants.items():
            summaries: dict[str, CurveSummary] = {}
            for name, group in animals.items():
                labels = [mouse for mouse, _ in group]
                stacked = np.vstack([pick(a) for _, a in group])
                summary = summarise_curves(stacked)
                summaries[name] = summary
                _save_figure(
                    mode_dir / f"{prefix}profile_{name}.png",
                    lambda ax: plot_condition(
                        name + (_axis_note(group_axis[name]) if mode == "pca1" else ""),
                        stacked, labels, summary, ax, ylabel,
                    ),
                    unity_line=prefix == "normalised_",
                )
            _save_figure(
                mode_dir / f"{prefix}profile_condition_means.png",
                lambda ax: plot_means(summaries, ax, ylabel),
                unity_line=prefix == "normalised_",
            )

        for name, group in animals.items():
            for mouse, a in group:
                rows.append(pd.DataFrame({
                    "mode": mode, "condition": name, "mouse": mouse,
                    "atlas_position": _grid_x(), "value": a.tissue[mode],
                    "atlas_value": a.atlas[mode], "normalised": normalise(a, mode),
                    "normalised_by": norm_tag,
                }))

    pd.concat(rows, ignore_index=True).to_csv(out_dir / "aligned_profiles.csv", index=False)


def _shift_note(animal: AlignedAnimal) -> str:
    if animal.shift == animal.correlation_shift:
        return f"{animal.shift}"
    return f"{animal.shift} (plain correlation {animal.correlation_shift})"


if __name__ == "__main__":
    main()
