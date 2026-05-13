# ================================================================
# 0. Section: IMPORTS
# ================================================================
import os

import pandas as pd

from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from matplotlib.figure import Figure


# ================================================================
# 1. Section: Functions
# ================================================================
@dataclass
class SaveSettings:
    name: str = f"plot_{datetime.today()}"
    format: list[str] = field(default_factory=lambda: ["svg", "pdf"])
    out_path: Path = Path("out/")

    def __post_init__(self):
        os.makedirs(self.out_path, exist_ok=True)

    def save_plot(self, fig: Figure) -> None:
        base_path = self.out_path / self.name
        for form in self.format:
            out_path = base_path.with_suffix(f".{form}")
            fig.savefig(out_path, bbox_inches="tight")

    def save_data(
        self, data: pd.DataFrame | list[pd.DataFrame], sheet_names: list[str] = []
    ) -> None:
        # 0. Initialize the sheet_names properly
        if len(sheet_names) < 1 and isinstance(data, list):
            sheet_names = [f"data_{str(p)}" for p in range(len(data))]

        # A. Make sure that it agrees with the size of data
        if isinstance(data, list) and len(data) != len(sheet_names):
            raise IndexError(
                "The sheet names and the amount of data need to be of same length"
            )

        # 1. Initialize the path
        base_path = self.out_path / f"{self.name}_data.xlsx"

        # 2. Writes the data
        with pd.ExcelWriter(base_path, engine="openpyxl") as writer:
            if isinstance(data, pd.DataFrame):
                data.to_excel(writer, index=True)
            else:
                for idx, dt in enumerate(data):
                    dt.to_excel(writer, sheet_name=sheet_names[idx], index=True)
