# ================================================================
# 0. Section: IMPORTS
# ================================================================
import h5py
import json

from dataclasses import dataclass
from pathlib import Path

from .Source import Source
from ..domain_model.data import ClearPoints, Atlas, ClearVolume, SampleBatch



# ================================================================
# 1. Section: Functions
# ================================================================
@dataclass
class DataDownloader:
    source: Source


    def download_batch(self, batch: SampleBatch, pipeline_id: int, step: int) -> Path:
        step_path = self.source.step_path(pipeline_id, step)
        step_path.mkdir(parents=True, exist_ok=True)

        self._download_tissue(batch.tissue, pipeline_id, step)
        self._download_atlas(batch.atlas, pipeline_id, step)
        self._download_cells(batch.cells, pipeline_id, step)

        return self.source.step_path(pipeline_id, step)



    def _download_tissue(
        self, tissue: ClearVolume, pipeline_id: int, step: int,
    ) -> Path:
        step_path = self.source.step_path(pipeline_id, step)
        file_path = step_path / f"{self.source.tissue_base_name}.hdf5"

        with h5py.File(file_path, "w") as f:
            f.create_dataset(
                "data",
                data=tissue.data,
                compression="gzip",
                compression_opts=4,
            )

            f.attrs["resolution"] = json.dumps(tissue.resolution)
            f.attrs["unit"] = json.dumps(tissue.unit)
            f.attrs["orientation"] = tissue.orientation
            f.attrs["tissue_type"] = tissue.tissue_type.name

        return file_path


    def _download_atlas(
        self, atlas: Atlas, pipeline_id: int, step: int
    ) -> Path:
        step_path = self.source.step_path(pipeline_id, step)
        file_path = step_path / f"{self.source.atlas_name}.hdf5"

        with h5py.File(file_path, "w") as f:
            f.create_dataset(
                "data",
                data=atlas.data,
                compression="gzip",
                compression_opts=4,
            )

            f.create_dataset(
                "hemisphere",
                data=atlas.hemisphere,
                compression="gzip",
                compression_opts=4,
            )

            look_up_json = atlas.look_up.to_json(orient="records")

            f.create_dataset(
                "look_up",
                data=look_up_json,
                dtype=h5py.string_dtype(encoding="utf-8"),
            )

            f.attrs["resolution"] = json.dumps(atlas.resolution)
            f.attrs["unit"] = json.dumps(atlas.unit)
            f.attrs["orientation"] = atlas.orientation
            f.attrs["tissue_type"] = atlas.tissue_type.name
            f.attrs["class"] = atlas.__class__.__name__
            f.attrs["look_up_format"] = "json_records"

        return file_path


    def _download_cells(
        self, cells: ClearPoints | ClearVolume, pipeline_id: int, step: int
    ) -> Path:
        step_path = self.source.step_path(pipeline_id, step)
        file_path = step_path / f"{self.source.cells_base_name}.hdf5"

        with h5py.File(file_path, "w") as f:
            f.create_dataset(
                "data",
                data=cells.data,
                compression="gzip",
                compression_opts=4,
            )

            f.attrs["resolution"] = json.dumps(cells.resolution)
            f.attrs["unit"] = json.dumps(cells.unit)
            f.attrs["orientation"] = cells.orientation
            f.attrs["tissue_type"] = cells.tissue_type.name
            f.attrs["class"] = cells.__class__.__name__

        return file_path
