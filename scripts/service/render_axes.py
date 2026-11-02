"""Render a loaded sample batch as one GIF per volume axis.

This service script loads a single mouse/pipeline/step batch through the
existing ``ClearTissueProject`` API and renders it as three GIFs — one per
volume axis — sweeping every slice along that axis. Any combination of the
four data modalities (tissue, cells, atlas, atlas tissue) can be shown at
once; when more than one is selected each is composited over the next at
alpha ``1/n`` in the order given, so the last modality listed sits on top.

Frames are written both as individual PNGs and as a compiled GIF, under a
``renders/`` folder. The rendering itself is pure (array in, array out); all
file writes are explicit calls from ``main`` so nothing is produced as a
hidden side effect of loading or blending.

Main components:
    RenderLayer: abstract one-modality-to-RGBA renderer; one subclass per
        data type (TissueLayer, CellsLayer, AtlasLayer, AtlasTissueLayer).
    blend_axis_frames: pure slice-sweep + alpha compositing for one axis.
    main: CLI entry point wiring load -> render -> save.
"""

# ================================================================
# 0. Section: IMPORTS
# ================================================================
import argparse

from abc import abstractmethod
from pathlib import Path
from typing import ClassVar
from dataclasses import dataclass, field

import numpy as np

from numpy.typing import NDArray
from scipy.ndimage import zoom
from PIL import Image
from tqdm import tqdm

from cleartissue.domain_model.data import (
    SampleBatch,
    ClearPoints,
    ClearVolume,
    TissueType,
)
from cleartissue.service.ClearTissueProject import ClearTissueProject
from cleartissue.domain_model.transformations.utils.points_to_volume import (
    get_points_as_volume,
)

# ================================================================
# 1. Section: Constants
# ================================================================
# Low/high percentiles for intensity normalisation, over nonzero voxels only.
_INTENSITY_PERCENTILES: tuple[float, float] = (1.0, 99.5)

# Seed for the atlas region colour lookup table, so colours are stable.
_ATLAS_COLOUR_SEED: int = 7

# Registry of selectable modalities: CLI name -> concrete layer class.
_LAYER_REGISTRY: dict[str, type["RenderLayer"]] = {}


# ================================================================
# 2. Section: Render layers
# ================================================================
@dataclass
class RenderLayer:
    """One data modality, resampled to a shared shape and rendered to RGBA.

    A layer owns a 3D volume already resampled to the common render shape.
    Concrete subclasses define how to pull their raw volume out of a batch
    (``extract``) and how to colour a single 2D slice (``colorize``). The
    class-level ``name`` is the discriminator used by the ``--data`` CLI flag
    and the registry; ``is_label`` switches resampling to nearest-neighbour so
    integer region ids are preserved.

    Attributes:
        volume: 3D source array already resampled to the common render shape.
    """

    name: ClassVar[str]
    is_label: ClassVar[bool] = False

    volume: NDArray
    _vmin: float = field(init=False, default=0.0)
    _vmax: float = field(init=False, default=1.0)

    def __init_subclass__(cls, **kwargs) -> None:
        super().__init_subclass__(**kwargs)
        if getattr(cls, "name", None):
            _LAYER_REGISTRY[cls.name] = cls

    def __post_init__(self) -> None:
        self._vmin, self._vmax = _nonzero_percentiles(self.volume)

    @classmethod
    @abstractmethod
    def extract(cls, batch: SampleBatch) -> NDArray:
        """Return this modality's raw 3D volume from a batch.

        Implementations must return a 3D array in the batch's native shape;
        resampling to the common render shape happens afterwards in
        ``build_layers``.
        """
        raise NotImplementedError

    @abstractmethod
    def colorize(self, slice_2d: NDArray) -> NDArray:
        """Return an ``(H, W, 4)`` float RGBA image in ``[0, 1]`` for a slice.

        Implementations must put signal strength in the alpha channel, so the
        caller can composite layers without knowing their contents.
        """
        raise NotImplementedError


@dataclass
class TissueLayer(RenderLayer):
    """Cleared-tissue intensity, rendered as a near-white signal."""

    name: ClassVar[str] = "tissue"
    colour: ClassVar[tuple[float, float, float]] = (0.92, 0.92, 0.92)

    @classmethod
    def extract(cls, batch: SampleBatch) -> NDArray:
        return np.asarray(batch.tissue.data, dtype=np.float32)

    def colorize(self, slice_2d: NDArray) -> NDArray:
        return _intensity_rgba(slice_2d, self.colour, self._vmin, self._vmax)


@dataclass
class CellsLayer(RenderLayer):
    """Detected cells, rendered in red; points are rasterised to a volume."""

    name: ClassVar[str] = "cells"
    colour: ClassVar[tuple[float, float, float]] = (0.976, 0.322, 0.357)

    @classmethod
    def extract(cls, batch: SampleBatch) -> NDArray:
        cells = batch.cells
        if isinstance(cells, ClearPoints):
            # Rasterise onto the tissue grid so cells share its coordinates.
            volume = get_points_as_volume(cells.data, batch.tissue.shape)
            return np.asarray(volume, dtype=np.float32)
        if isinstance(cells, ClearVolume):
            return np.asarray(cells.data, dtype=np.float32)
        raise TypeError(f"Unsupported cells type: {type(cells).__name__}")

    def colorize(self, slice_2d: NDArray) -> NDArray:
        return _intensity_rgba(slice_2d, self.colour, self._vmin, self._vmax)


@dataclass
class AtlasLayer(RenderLayer):
    """Atlas annotation, rendered with a distinct colour per region id."""

    name: ClassVar[str] = "atlas"
    is_label: ClassVar[bool] = True

    _lut: NDArray = field(init=False)

    @classmethod
    def extract(cls, batch: SampleBatch) -> NDArray:
        return np.asarray(batch.atlas.data).astype(np.int32, copy=False)

    def __post_init__(self) -> None:
        super().__post_init__()
        self._lut = _region_colour_lut(self.volume)

    def colorize(self, slice_2d: NDArray) -> NDArray:
        labels = slice_2d.astype(np.int64, copy=False)
        rgba = np.zeros((*labels.shape, 4), dtype=np.float32)
        rgba[..., :3] = self._lut[labels]
        rgba[..., 3] = (labels > 0).astype(np.float32)
        return rgba


@dataclass
class AtlasTissueLayer(RenderLayer):
    """Atlas anatomical reference template, rendered as a blue signal."""

    name: ClassVar[str] = "atlas_tissue"
    colour: ClassVar[tuple[float, float, float]] = (0.60, 0.75, 1.0)

    @classmethod
    def extract(cls, batch: SampleBatch) -> NDArray:
        return np.asarray(batch.atlas.tissue_template, dtype=np.float32)

    def colorize(self, slice_2d: NDArray) -> NDArray:
        return _intensity_rgba(slice_2d, self.colour, self._vmin, self._vmax)


# ================================================================
# 3. Section: Rendering
# ================================================================
def build_layers(batch: SampleBatch, data_names: list[str]) -> list[RenderLayer]:
    """Build the selected layers, all resampled to a shared shape.

    The first selected modality defines the common render shape; every other
    modality is resampled to it (nearest-neighbour for label volumes, linear
    otherwise) so slices line up across layers.

    Args:
        data_names: modality names in render order; first is drawn at the
            bottom, last on top.

    Raises:
        KeyError: If a name is not a registered modality.
    """
    raws = {name: _layer_class(name).extract(batch) for name in data_names}
    target_shape = raws[data_names[0]].shape

    layers: list[RenderLayer] = []
    for name in data_names:
        cls = _layer_class(name)
        volume = _resample_to_shape(raws[name], target_shape, cls.is_label)
        layers.append(cls(volume=volume))

    return layers


def blend_axis_frames(
    layers: list[RenderLayer],
    axis: int,
    weights: list[float],
) -> list[NDArray]:
    """Sweep every slice along ``axis`` and composite the layers per slice.

    Each slice is composited back-to-front over black: every layer is drawn
    with its own alpha scaled by its blend ``weight``, so a weight of 1.0 lets
    a layer show at full opacity and a smaller weight fades it under the layers
    above. Leading and trailing slices with no signal in any layer are trimmed.

    Args:
        weights: per-layer blend weight in ``[0, 1]``, parallel to ``layers``.

    Returns:
        RGB ``uint8`` frames in sweep order; empty if every slice is blank.
    """
    n_slices = layers[0].volume.shape[axis]

    pairs: list[tuple[NDArray, bool]] = []
    for index in tqdm(range(n_slices), desc=f"axis {axis}"):
        pairs.append(_composite_slice(layers, axis, index, weights))

    return _trim_blank(pairs)


def save_pngs(frames: list[NDArray], out_dir: Path) -> None:
    """Write each frame as ``frame_XXXX.png`` into ``out_dir``."""
    out_dir.mkdir(parents=True, exist_ok=True)
    for index, frame in enumerate(frames):
        Image.fromarray(frame).save(out_dir / f"frame_{index:04d}.png")


def save_gif(frames: list[NDArray], out_path: Path, speed: float) -> None:
    """Compile frames into a looping GIF played at ``speed`` frames/second.

    Args:
        speed: playback rate in frames per second.
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)
    images = [Image.fromarray(frame) for frame in frames]
    duration_ms = max(1, round(1000.0 / speed))
    images[0].save(
        out_path,
        save_all=True,
        append_images=images[1:],
        duration=duration_ms,
        loop=0,
        disposal=2,
    )


# ──────────────────────────────────────────────────────
# 3.1 Subsection: Helper Functions
# ──────────────────────────────────────────────────────
def _layer_class(name: str) -> type[RenderLayer]:
    if name not in _LAYER_REGISTRY:
        known = ", ".join(sorted(_LAYER_REGISTRY))
        raise KeyError(f"Unknown data modality '{name}'. Choose from: {known}")
    return _LAYER_REGISTRY[name]


def _nonzero_percentiles(volume: NDArray) -> tuple[float, float]:
    nonzero = volume[volume > 0]
    if nonzero.size == 0:
        return 0.0, 1.0
    low, high = np.percentile(nonzero, _INTENSITY_PERCENTILES)
    if high <= low:  # flat volume — avoid a zero-width range
        high = low + 1.0
    return float(low), float(high)


def _intensity_rgba(
    slice_2d: NDArray,
    colour: tuple[float, float, float],
    vmin: float,
    vmax: float,
) -> NDArray:
    norm = np.clip((slice_2d - vmin) / (vmax - vmin), 0.0, 1.0)
    rgba = np.zeros((*slice_2d.shape, 4), dtype=np.float32)
    rgba[..., :3] = colour
    rgba[..., 3] = norm  # alpha carries signal strength
    return rgba


def _region_colour_lut(atlas: NDArray) -> NDArray:
    max_id = int(atlas.max()) if atlas.size else 0
    rng = np.random.default_rng(_ATLAS_COLOUR_SEED)
    lut = rng.uniform(0.25, 1.0, size=(max_id + 1, 3)).astype(np.float32)
    lut[0] = 0.0  # background id stays black
    return lut


def _resample_to_shape(
    volume: NDArray,
    target_shape: tuple[int, ...],
    is_label: bool,
) -> NDArray:
    if volume.shape == target_shape:
        return volume
    factors = np.asarray(target_shape, dtype=float) / np.asarray(volume.shape)
    order = 0 if is_label else 1
    return zoom(volume, factors, order=order)


def _composite_slice(
    layers: list[RenderLayer],
    axis: int,
    index: int,
    weights: list[float],
) -> tuple[NDArray, bool]:
    base_shape = np.take(layers[0].volume, index, axis=axis).shape
    acc_rgb = np.zeros((*base_shape, 3), dtype=np.float32)
    acc_alpha = np.zeros(base_shape, dtype=np.float32)

    for layer, weight in zip(layers, weights):
        rgba = layer.colorize(np.take(layer.volume, index, axis=axis))
        alpha = rgba[..., 3] * weight
        over = alpha[..., None]
        acc_rgb = rgba[..., :3] * over + acc_rgb * (1.0 - over)  # 'over' composite
        acc_alpha = alpha + acc_alpha * (1.0 - alpha)

    rgb = (np.clip(acc_rgb, 0.0, 1.0) * 255).astype(np.uint8)
    return rgb, bool(acc_alpha.max() > 0)


def _trim_blank(pairs: list[tuple[NDArray, bool]]) -> list[NDArray]:
    first = next((i for i, (_, content) in enumerate(pairs) if content), None)
    if first is None:
        return []
    last = next(i for i in range(len(pairs) - 1, -1, -1) if pairs[i][1])
    return [rgb for rgb, _ in pairs[first : last + 1]]


# ================================================================
# 4. Section: CLI
# ================================================================
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Render a mouse/pipeline/step batch as one GIF per axis.",
    )
    parser.add_argument("mouse_id", help="Mouse identifier, e.g. 189R")
    parser.add_argument("pipeline_id", type=int, help="Pipeline id, e.g. 2")
    parser.add_argument("step_id", type=int, help="Step id within the pipeline")
    parser.add_argument(
        "--tissue",
        default="sc",
        help="Tissue type: sc/spine/spinal_cord or br/brain (default: sc)",
    )
    parser.add_argument(
        "--data",
        nargs="+",
        default=["tissue", "cells"],
        choices=["tissue", "cells", "atlas", "atlas_tissue"],
        help="Modalities to render, bottom-to-top (default: tissue cells)",
    )
    parser.add_argument(
        "--alpha",
        nargs="+",
        type=float,
        default=None,
        help=(
            "Per-layer blend weight in [0, 1], parallel to --data (or one "
            "value for all). Default: 1/n for every layer. Use e.g. "
            "'--data atlas tissue --alpha 0.4 1.0' to fade a solid atlas "
            "under the tissue."
        ),
    )
    parser.add_argument(
        "--speed",
        type=float,
        default=20.0,
        help="GIF playback speed in frames per second (default: 20)",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("renders"),
        help="Root output folder (default: renders)",
    )
    return parser.parse_args()


def _resolve_weights(alpha: list[float] | None, n_layers: int) -> list[float]:
    """Expand the ``--alpha`` flag into one blend weight per layer.

    ``None`` falls back to ``1/n`` for every layer; a single value is applied
    to all; otherwise the count must match the number of selected modalities.

    Raises:
        ValueError: If the number of weights is neither 1 nor ``n_layers``.
    """
    if alpha is None:
        return [1.0 / n_layers] * n_layers
    if len(alpha) == 1:
        return alpha * n_layers
    if len(alpha) != n_layers:
        raise ValueError(f"--alpha expects 1 or {n_layers} values, got {len(alpha)}")
    return alpha


# ================================================================
# 5. Section: Main
# ================================================================
def main() -> None:
    args = parse_args()

    project = ClearTissueProject.load(
        mouse=args.mouse_id,
        tissue_type=TissueType.from_str(args.tissue),
    )
    batch = project.io.load_batch(pipeline_id=args.pipeline_id, step=args.step_id)

    layers = build_layers(batch, args.data)
    weights = _resolve_weights(args.alpha, len(args.data))

    run_dir = args.out_dir / f"{args.mouse_id}_p{args.pipeline_id:02d}_s{args.step_id}"
    modalities = "+".join(args.data)

    for axis in range(3):
        frames = blend_axis_frames(layers, axis, weights)
        if not frames:
            print(f"axis {axis}: no signal, skipped")
            continue

        png_dir = run_dir / f"axis{axis}_frames"
        gif_path = run_dir / f"axis{axis}.gif"

        save_pngs(frames, png_dir)
        save_gif(frames, gif_path, args.speed)

        print(f"axis {axis}: {len(frames)} frames -> {gif_path}")

    print(f"Rendered {modalities} for {args.mouse_id} into {run_dir}")


if __name__ == "__main__":
    main()
