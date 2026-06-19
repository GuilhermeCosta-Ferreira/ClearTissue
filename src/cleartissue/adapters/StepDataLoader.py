# ================================================================
# 0. Section: IMPORTS
# ================================================================
import io
import h5py
import json

import numpy as np
import pandas as pd

from dataclasses import dataclass
from numpy.typing import NDArray

from .Source import Source
from ..domain_model.data import ClearPoints, Atlas, ClearVolume, TissueType



# ================================================================
# 1. Section: Functions
# ================================================================
@dataclass
class StepDataLoader:
    source: Source

    def load_cells(self, pipeline_id: int, step_id: int) -> ClearPoints | ClearVolume:
        step_path = self.source.step_path(pipeline_id, step_id)
        file_path = step_path / f"{self.source.cells_base_name}.hdf5"

        with h5py.File(file_path, "r") as f:
            data: NDArray = np.asarray(f["data"])

            resolution = tuple(json.loads(f.attrs["resolution"]))  # type: ignore
            unit = tuple(json.loads(f.attrs["unit"]))  # type: ignore
            orientation = str(f.attrs["orientation"])
            tissue_type = TissueType[str(f.attrs["tissue_type"])]
            class_name = str(f.attrs["class_name"])

        if class_name == "ClearPoints":
            return ClearPoints(
                data=data,
                resolution=resolution,
                unit=unit,
                orientation=orientation,
                tissue_type=tissue_type,
            )
        elif class_name == "ClearVolume":
            return ClearVolume(
                data=data,
                resolution=resolution,
                unit=unit,
                orientation=orientation,
                tissue_type=tissue_type,
            )
        else:
            raise ValueError(f"Unknown class name: {class_name}")


    def load_tissue(self, pipeline_id: int, step_id: int) -> ClearVolume:
        step_path = self.source.step_path(pipeline_id, step_id)
        file_path = step_path / f"{self.source.tissue_base_name}.hdf5"

        with h5py.File(file_path, "r") as f:
            data: NDArray = np.asarray(f["data"])

            resolution = tuple(json.loads(f.attrs["resolution"]))  # type: ignore
            unit = tuple(json.loads(f.attrs["unit"]))  # type: ignore
            orientation = str(f.attrs["orientation"])
            tissue_type = TissueType[str(f.attrs["tissue_type"])]

        return ClearVolume(
            data=data,
            resolution=resolution,
            unit=unit,
            orientation=orientation,
            tissue_type=tissue_type,
        )


    def load_atlas(self, pipeline_id: int, step_id: int) -> Atlas:
        step_path = self.source.step_path(pipeline_id, step_id)
        file_path = step_path / f"{self.source.atlas_name}.hdf5"
        with h5py.File(file_path, "r") as f:
            look_up_json = f["look_up"].asstr()[()]  # type: ignore

            return Atlas(
                data = np.asarray(f["data"]),
                hemisphere = np.asarray(f["hemisphere"]),
                look_up = pd.read_json(io.StringIO(look_up_json), orient="records"),  # type: ignore
                resolution = tuple(json.loads(f.attrs["resolution"])),  # type: ignore
                unit = tuple(json.loads(f.attrs["unit"])),  # type: ignore
                orientation = str(f.attrs["orientation"]),
                tissue_type = TissueType[str(f.attrs["tissue_type"])],
            )
