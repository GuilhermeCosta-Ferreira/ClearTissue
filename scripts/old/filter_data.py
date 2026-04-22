# ================================================================
# 0. Section: IMPORTS
# ================================================================
from matplotlib import pyplot as plt

from pathlib import Path

from clearbrain import (
    load_points,
    plot_3d_clear_points
)
from clearbrain.preprocess import(
    scale_points,
    filter_low_density_points,
)
from clearbrain.save import save_to_json



# ================================================================
# 1. Section: INPUTS
# ================================================================
# IO Settings
DATA_FOLDER: Path = Path("data")
PLOT_FOLDER: Path = Path("out")

MICE: list = ["32B"]
FILE_TARGET: str = "raw_points_sc.json"

# Scaling Settings
SCALING: tuple[float, float, float] = (2.22, 1.0, 1.0)

# Filtering Settings
DENSITY_RADIUS: int = 50
MIN_DENSITY: int = 25 #20 [20, 50[, [20, 30]

# Plotting Settings
PLOT_SUBSAMPLE: int = 80  # Get's every X points



# ================================================================
# 2. Section: MAIN
# ================================================================
if __name__ == "__main__":
    for mouse in MICE:
        filepath = DATA_FOLDER / mouse / FILE_TARGET

        # 1. Load the points
        points = load_points(filepath)
        points = scale_points(points, SCALING)

        # 2. Remove sparse points
        points = filter_low_density_points(points, DENSITY_RADIUS, MIN_DENSITY)

        # 4. Generate the 3D plot
        plot_3d_clear_points(points, PLOT_SUBSAMPLE)
        plt.show()

        # 5. Saved the data
        out_path = save_to_json(points.tolist(), filepath.parent, "filtered_points_sc.json")
        print(f"Saved filtered data from {mouse} into {out_path}")
