# ================================================================
# 0. Section: IMPORTS
# ================================================================
from dataclasses import dataclass

from .Source import Source
from ..domain_model.config import PipelineConfig
from .load import load_yaml


# ================================================================
# 1. Section: Functions
# ================================================================
@dataclass
class PipelineConfigRepository:
    source: Source

    def load_config(self, pipeline_id: int) -> PipelineConfig:
        config_path = self.source.pipeline_config_path(pipeline_id)
        yamls_dict = load_yaml(config_path)

        return PipelineConfig(config=yamls_dict)


    def update_config(self, pipeline_id: int, config: PipelineConfig) -> None:
        pass
