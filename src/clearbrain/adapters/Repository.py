# ================================================================
# 0. Section: IMPORTS
# ================================================================
from pathlib import Path
from ruamel.yaml import YAML
from dataclasses import dataclass

from .Source import Source
from .download import download_json
from .utils import standard_numeric_id



# ================================================================
# 1. Section: Functions
# ================================================================
@dataclass
class Repository:
    source: Source

    _pipeline_base_name: str = "pipeline"
    _config_path: Path = Path("src/clearbrain/adapters/configs/base.yaml")



    def init_project(self) -> None:
        # 1. Check if project folder already exists
        if self.source.folder_path.exists():
            raise FileExistsError(f"Project folder already exists: {self.source.folder_path}")

        # 2. Create project folder
        self.source.folder_path.mkdir(parents=True, exist_ok=True)

        # 3. Create raw data folder
        raw = self.source.folder_path / "raw"
        raw.mkdir(parents=True, exist_ok=True)

        # 4. Create metadata file
        metadata = build_metadata(self.source)
        metadata_path = self.source.folder_path / "metadata.json"
        download_json(metadata, metadata_path, False)

    def init_new_pipeline(self, pipeline_id: int, pipeline_name: str = "") -> str:
        # 1. Creates the folder for the pipeline
        pipeline_id_str = standard_numeric_id(pipeline_id, 2)
        pipeline_name = (
            pipeline_name
            if pipeline_name
            else f"{self._pipeline_base_name}_{pipeline_id_str}"
        )
        path = self.source.folder_path / f"{self._pipeline_base_name}_{pipeline_id_str}"
        path.mkdir(parents=True, exist_ok=True)

        # 2. Starts the yaml parser
        yaml = YAML()
        yaml.preserve_quotes = True
        yaml.indent(mapping=2, sequence=4, offset=2)

        # 3. Loads the default yaml
        with open(self._config_path, "r") as f:
            config = yaml.load(f)

        # 4. Updates the config with source information
        config["mouse"] = self.source.mouse
        config["tissue_type"] = self.source.tissue_type.as_str
        config["pipeline_id"] = pipeline_id
        config["pipeline_name"] = pipeline_name

        # 5. Saves the updated config to the pipeline folder
        with open(path / "config.yaml", "w") as f:
            yaml.dump(config, f)

        return pipeline_name

    def get_built_pipeline_ids(self) -> list[int]:
        project_dir = self.source.folder_path
        pipeline_dirs = project_dir.glob(f"{self._pipeline_base_name}_*")
        return [int(dir.name.split("_")[-1]) for dir in pipeline_dirs]

    def get_available_id(self) -> int:
        used_ids = self.get_built_pipeline_ids()
        return max(used_ids) + 1 if used_ids else 1

def build_metadata(source: Source) -> dict:
    metadata = source.as_dict

    metadata["cells_resolution"] = (4.8, 1.8, 1.8)
    metadata["cells_unit"] = ("um", "um", "um")
    metadata["cells_orientation"] = "psl"
    metadata["atlas_unit"] = ("um", "um", "um")

    return metadata
