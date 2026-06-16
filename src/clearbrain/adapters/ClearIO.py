# ================================================================
# 0. Section: IMPORTS
# ================================================================
from dataclasses import dataclass
from pathlib import Path

from .Source import Source
from .DataLoader import DataLoader
from .DataDownloader import DataDownloader
from ..domain_model.data import SampleBatch



# ================================================================
# 1. Section: Functions
# ================================================================
@dataclass
class ClearIO:
    source: Source

    def __post_init__(self):
        self._loader = DataLoader(self.source)
        self._downloader = DataDownloader(self.source)

    def load_raw(self) -> SampleBatch:
        return self._loader.load_raw()

    def load_batch(self, pipeline_id: int, step: int) -> SampleBatch:
        return self._loader.load_batch(pipeline_id, step)

    def download_batch(self, batch: SampleBatch, pipeline_id: int, step: int) -> Path:
        return self._downloader.download_batch(batch, pipeline_id, step)
