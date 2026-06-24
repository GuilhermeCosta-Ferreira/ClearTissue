# ================================================================
# 0. Section: IMPORTS
# ================================================================
import trimesh
import colorsys

import numpy as np

from tqdm import tqdm
from pathlib import Path
from skimage import measure
from numpy.typing import NDArray
from dataclasses import dataclass
from scipy.ndimage import binary_erosion
from trimesh.visual import ColorVisuals

from .Source import Source
from ..domain_model.data import SampleBatch, ClearVolume, ClearPoints, Atlas



# ================================================================
# 1. Section: Functions
# ================================================================
@dataclass
class HDF5_to_Ply:
    source: Source

    def convert_batch(self, batch: SampleBatch, step_path: Path) -> list[Path]:
        out_paths: list[Path] = []

        tissue_ply = self._convert_volume(batch.tissue, step_path)
        out_paths.append(tissue_ply)

        if isinstance(batch.cells, ClearVolume):
            ply_path = step_path / f"{self.source.cells_base_name}.ply"
            cells_ply = self._convert_volume(batch.cells, step_path, ply_path, 0.9)
            out_paths.append(cells_ply)
        elif isinstance(batch.cells, ClearPoints):
            print("Cells are not a ClearVolume, so will not be converted to Ply")

        atlas_ply = self._convert_atlas(batch.atlas, step_path)
        out_paths.extend(atlas_ply)

        return out_paths


    # ──────────────────────────────────────────────────────
    # 1.1 Subsection: Helper Functions
    # ──────────────────────────────────────────────────────
    def _convert_volume(
        self,
        tissue: ClearVolume,
        step_path: Path,
        ply_path: Path | None = None,
        level: float = 0.0
    ) -> Path:
        ply_path = step_path / f"{self.source.tissue_base_name}.ply" if ply_path is None else ply_path

        data = tissue.data

        mesh = build_mesh(data, level)
        mesh.export(ply_path)

        return ply_path

    def _convert_atlas(
        self,
        atlas: Atlas,
        step_path: Path,
    ) -> list[Path]:
        atlas_base_path = step_path / "atlas_parts"
        atlas_base_path.mkdir(parents=True, exist_ok=True)

        erode_struct = np.ones((2, 2, 2))
        smaller_struct = np.ones((1, 1, 1))

        out_paths: list[Path] = []
        separated_atlas = separate_brain_parts(atlas.data)

        total_parts = len(separated_atlas)

        for index, part in tqdm(enumerate(separated_atlas)):
            binary = np.where(part > 0, 1, 0)
            eroded_binary = binary_erosion(binary, structure=erode_struct)

            if np.sum(eroded_binary) <= 10:
                eroded_binary = binary_erosion(binary, structure=smaller_struct)

            if np.sum(eroded_binary) <= 10:
                eroded_binary = binary.copy()

            part_id = get_part_id(part)

            mesh = build_mesh(eroded_binary, level=0.5)

            color = part_color_from_index(index, total_parts)
            mesh = apply_vertex_color(mesh, color)

            part_path = atlas_base_path / f"{self.source.atlas_name}_part_{part_id}.ply"
            mesh.export(part_path)

            out_paths.append(part_path)

        return out_paths



def build_mesh(data: NDArray, level: float) -> trimesh.Trimesh:
    verts, faces, normals, values = measure.marching_cubes(data, level=level)

    mesh = trimesh.Trimesh(
        vertices=verts,
        faces=faces,
        vertex_normals=normals,
        process=True,
    )

    return mesh

def separate_brain_parts(brain_labels: np.ndarray) -> NDArray:
    brain_parts = []
    for label in np.unique(brain_labels):
        if(label == 0):
            continue  # Skip background
        part = np.where(brain_labels == label, label, 0)
        brain_parts.append(part)
    return np.array(brain_parts)

def get_part_id(volume: NDArray) -> int:
    ids = np.unique(volume)
    return int(ids[-1])

def part_color_from_index(index: int, total: int, alpha: int = 255) -> NDArray:
    hue = index / max(total, 1)
    saturation = 0.85
    value = 1.0

    r, g, b = colorsys.hsv_to_rgb(hue, saturation, value)

    return np.array(
        [
            int(r * 255),
            int(g * 255),
            int(b * 255),
            alpha,
        ],
        dtype=np.uint8,
    )


def apply_vertex_color(mesh: trimesh.Trimesh, color: NDArray) -> trimesh.Trimesh:
    vertex_colors = np.tile(color, (len(mesh.vertices), 1))

    if not isinstance(mesh.visual, ColorVisuals):
        raise ValueError(f"Visual from mesh was indeed not ColorVisuals, it was: {mesh.visual.__class__}")

    mesh.visual.vertex_colors = vertex_colors
    return mesh
