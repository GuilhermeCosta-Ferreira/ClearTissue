# ================================================================
# 0. Section: IMPORTS
# ================================================================
from dataclasses import dataclass
import json
import os
from pathlib import Path

from ..tissue import TissueType



# ================================================================
# 1. Section: Functions
# ================================================================
@dataclass(frozen=True)
class Metadata:
    mouse: str
    tissue_type: TissueType
    file_path: Path
    description: str = ""



    # ================================================================
    # 3. Section: Properties
    # ================================================================
    @property
    def dict(self) -> dict:
        return {
            "mouse": self.mouse,
            "tissue_type": self.tissue_type.str,
            "description": self.description
        }



    # ================================================================
    # 4. Section: Functions
    # ================================================================
    def create_metadata(self, path: Path) -> None:
        if os.path.exists(path):
            print(f"Metada file already exists ({str(path)}), if you want to update, run the update function")
            return

        with path.open("w", encoding="utf-8") as f:
            json.dump(self.dict, f, indent=2)
