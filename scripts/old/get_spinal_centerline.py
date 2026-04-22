# ================================================================
# 0. Section: IMPORTS
# ================================================================
from matplotlib import pyplot as plt

from pathlib import Path

from clearbrain.save import save_to_json
from clearbrain.centerline import (
    get_centerline,
    smooth_centerline,
    add_centerline
)

from clearbrain import (
    load_points,
    plot_3d_clear_points
)



# ================================================================
# 1. Section: INPUTS
# ================================================================
# IO Settings
DATA_FOLDER: Path = Path("data")
MICE: list = ["32B"]
FILE_TARGET: str = "filtered_points_sc.json"

# Cenetrline Settings
BIN_WIDTH: int = 500
HIGHLIGHT_CENTERLINE: bool = True # makes sure the line is drawn on top of it

# Centerline Smoothing Settings
SPLINE_SMOOTHING: float = 5000.0  # ← extremely smooth (as requested)
N_POINTS_ON_LINE: float = 4000  # more points = perfectly smooth visual
PLOT_SUBSAMPLE: int = 80  # Get's every X points



# ================================================================
# 2. Section: MAIN
# ================================================================
if __name__ == "__main__":
    for mouse in MICE:
        filepath = DATA_FOLDER / mouse / FILE_TARGET

        # 1. Load the points
        points = load_points(filepath)

        # 2. Get the centerline and smooths it
        centerline = get_centerline(points, BIN_WIDTH)
        centerline = smooth_centerline(centerline, SPLINE_SMOOTHING, N_POINTS_ON_LINE)

        # 3. Generate the 3D plot
        fig, ax = plot_3d_clear_points(points, PLOT_SUBSAMPLE)
        add_centerline(ax, centerline, HIGHLIGHT_CENTERLINE)
        plt.show()

        # 4. Saved the data
        out_path = save_to_json(centerline.tolist(), filepath.parent, "centerline_sc.json")
        print(f"Saved filtered data from {mouse} into {out_path}")
