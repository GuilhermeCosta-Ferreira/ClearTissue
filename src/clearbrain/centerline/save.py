# ================================================================
# 0. Section: IMPORTS
# ================================================================
import numpy as np

import plotly.graph_objects as go
from pathlib import Path



# ================================================================
# 1. Section: Functions
# ================================================================
def save_centerline(
    centerline: np.ndarray,
    fig: go.Figure,
    output_folder: Path,
    sample_name: str,
) -> None:
    # 1. Gets the paths
    npy_path = output_folder / f"{sample_name}_spinal_cord_centerline.npy"
    html_path = output_folder / f"{sample_name}_spinal_cord_centerline.html"

    # 2. Saves the data as npy
    np.save(npy_path, centerline)
    print(f" ✅ .npy saved: {npy_path}")

    # 3. Saves the data as HTML
    fig.write_html(html_path)
    print(f" ✅ 3D HTML saved: {html_path}\n")
