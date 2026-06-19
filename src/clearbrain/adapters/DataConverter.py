# ================================================================
# 0. Section: IMPORTS
# ================================================================
from dataclasses import dataclass
from pathlib import Path

from .Source import Source
from .HDF5_to_NII import HDF5_to_Nii
from .DataLoader import DataLoader



# ================================================================
# 1. Section: Functions
# ================================================================
@dataclass
class DataConverter:
    source: Source

    def __post_init__(self):
        self.loader = DataLoader(self.source)
        self.nii_converter = HDF5_to_Nii(self.source)

    def convert_batch(self, pipeline_id: int, step_id: int, out_file_type: str = ".nii.gz") -> list[Path]:
        batch = self.loader.load_batch(pipeline_id, step_id)
        step_path = self.source.step_path(pipeline_id, step_id)

        if out_file_type == ".nii.gz":
            return self.nii_converter.convert_batch(batch, step_path)
        else:
            raise ValueError(f"Unsupported file type: {out_file_type}")
