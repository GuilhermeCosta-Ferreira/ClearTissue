# ================================================================
# 0. Section: IMPORTS
# ================================================================
from pathlib import Path
from dataclasses import dataclass
from brainglobe_atlasapi.atlas_name import AtlasName

from ..domain_model.data import TissueType
from .utils import standard_numeric_id



# ================================================================
# 0. Section: IMPORTS
# ================================================================
@dataclass(frozen=True, slots=True)
class Source:
    mouse: str
    tissue_type: TissueType
    base_path: Path
    atlas_name: AtlasName

    @property
    def folder_path(self) -> Path:
        return self.base_path / self.mouse

    @property
    def file_base_name(self) -> str:
        return f"{self.tissue_type.as_str}_sample"

    @property
    def as_dict(self) -> dict:
        return {
            "mouse": self.mouse,
            "tissue_type": self.tissue_type.as_str,
        }

    @property
    def raw_path(self) -> Path:
        return self.folder_path / "raw"

    @property
    def cells_base_name(self) -> str:
        return f"{self.file_base_name}_cells"

    @property
    def tissue_base_name(self) -> str:
        return f"{self.file_base_name}_tissue"

    @property
    def metadata_path(self) -> Path:
        return self.folder_path / "metadata.json"

    def pipeline_path(self, pipeline_id: int) -> Path:
        return self.folder_path / f"pipeline_{standard_numeric_id(pipeline_id, 2)}"

    def pipeline_config_path(self, pipeline_id: int) -> Path:
        return self.pipeline_path(pipeline_id) / "config.yaml"

    def step_path(self, pipeline_id: int, step: int) -> Path:
        return self.pipeline_path(pipeline_id) / f"step_{step}"
