# ================================================================
# 0. Section: IMPORTS
# ================================================================
from dataclasses import dataclass
from pathlib import Path

from .Source import Source
from ..domain_model.data import SampleBatch, ClearVolume, ClearPoints, Atlas
from .utils import affine_from_attrs, save_nii_gz



# ================================================================
# 1. Section: Functions
# ================================================================
@dataclass
class HDF5_to_Nii:
    source: Source

    def convert_batch(self, batch: SampleBatch, step_path: Path) -> list[Path]:
        out_paths: list[Path] = []

        tissue_nii = self._convert_volume(batch.tissue, step_path)
        out_paths.append(tissue_nii)

        if isinstance(batch.cells, ClearVolume):
            nii_path = step_path / f"{self.source.cells_base_name}.nii.gz"
            cells_nii = self._convert_volume(batch.cells, step_path, nii_path)
            out_paths.append(cells_nii)
        elif isinstance(batch.cells, ClearPoints):
            print("Cells are not a ClearVolume, so will not be converted to NII")

        atlas_nii = self._convert_atlas(batch.atlas, step_path)
        out_paths.extend(atlas_nii)

        return out_paths


    # ──────────────────────────────────────────────────────
    # 1.1 Subsection: Helper Functions
    # ──────────────────────────────────────────────────────
    def _convert_volume(
        self,
        tissue: ClearVolume,
        step_path: Path,
        nii_path: Path | None = None
    ) -> Path:
        nii_path = step_path / f"{self.source.tissue_base_name}.nii.gz" if nii_path is None else nii_path

        data = tissue.data
        affine = affine_from_attrs(tissue)

        save_nii_gz(data, affine, nii_path)

        return nii_path

    def _convert_atlas(
        self,
        atlas: Atlas,
        step_path: Path,
    ) -> list[Path]:
        atlas_path = step_path / f"{self.source.atlas_name}.nii.gz"
        hemisphere_path = step_path / f"{self.source.atlas_name}_hemisphere.nii.gz"

        out_paths: list[Path] = []

        affine = affine_from_attrs(atlas)

        atlas_data = atlas.data
        save_nii_gz(atlas_data, affine, atlas_path)
        out_paths.append(atlas_path)

        hemisphere = atlas.hemisphere
        save_nii_gz(hemisphere, affine, hemisphere_path)
        out_paths.append(hemisphere_path)

        return out_paths
