# ================================================================
# 0. Section: IMPORTS
# ================================================================
from dataclasses import dataclass

from .Source import Source
from .load import load_json
from .RawDataLoader import RawDataLoader
from .StepDataLoader import StepDataLoader
from ..domain_model.data import SampleBatch



# ================================================================
# 1. Section: Functions
# ================================================================
@dataclass
class DataLoader:
    source: Source



    # ================================================================
    # 2. Section: Post-Initialization
    # ================================================================
    def __post_init__(self):
        self._raw_loader = RawDataLoader(self.source)
        self._step_loader = StepDataLoader(self.source)



    # ================================================================
    # 3. Section: Methods
    # ================================================================
    def load_metadata(self) -> dict:
        return load_json(self.source.metadata_path)  # type: ignore

    def load_raw(self) -> SampleBatch:
        metadata = self.load_metadata()
        cells = self._raw_loader.load_raw_cells(
            metadata["cells_resolution"],
            metadata["cells_unit"],
            metadata["cells_orientation"]
        )
        atlas = self._raw_loader.load_raw_atlas(metadata["atlas_unit"])
        tissue = self._raw_loader.load_raw_tissue()

        if tissue.tissue_type != self.source.tissue_type:
            raise ValueError("Tissue type in metadata does not match tissue type in H5 file")

        return SampleBatch(cells=cells, atlas=atlas, tissue=tissue)

    def load_batch(self, pipeline_id: int, step: int) -> SampleBatch:
        cells = self._step_loader.load_cells(pipeline_id, step)
        tissue = self._step_loader.load_tissue(pipeline_id, step)
        atlas = self._step_loader.load_atlas(pipeline_id, step)

        return SampleBatch(cells=cells, atlas=atlas, tissue=tissue)
