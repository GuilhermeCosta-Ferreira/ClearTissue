# ================================================================
# 0. Section: IMPORTS
# ================================================================
from matplotlib import pyplot as plt

from pathlib import Path

from clearbrain.sections import get_spinal_sections, add_spinal_sections
from clearbrain.centerline import add_centerline
from clearbrain.save import save_to_json
from clearbrain import load_points, plot_3d_clear_points



# ================================================================
# 1. Section: INPUTS
# ================================================================
# IO Settings
DATA_FOLDER: Path = Path("data")
MICE: list = ["32B"]
FILE_TARGET: str = "filtered_points_sc.json"

PRISM_HALF_WIDTH: int = 1000        # ← 2000×2000 square base (as requested)
PRISM_HALF_THICKNESS: int = 250     # ← 500 units thick
N_CUTS: int = 10                    # 10 axial cuts

HIGHLIGHT_CENTERLINE: bool = True # makes sure the line is drawn on top of it
PLOT_SUBSAMPLE: int = 80  # Get's every X points



# ================================================================
# 3. Section: MAIN
# ================================================================
if __name__ == '__main__':
    for mouse in MICE:
        filepath = DATA_FOLDER / mouse / FILE_TARGET
        centerline_path = DATA_FOLDER / mouse / "centerline_sc.json"

        # 1. Load the points
        points = load_points(filepath)
        centerline = load_points(centerline_path)

        # 2. Get the spinal sections
        spinal_sections, section_points, section_centers = get_spinal_sections(
            points,
            centerline,
            N_CUTS,
            PRISM_HALF_WIDTH,
            PRISM_HALF_THICKNESS
        )

        # 3. Generate the 3D plot
        fig, ax = plot_3d_clear_points(points, PLOT_SUBSAMPLE)
        ax = add_centerline(ax, centerline, HIGHLIGHT_CENTERLINE)
        add_spinal_sections(ax, spinal_sections)
        plt.show()


        # 4. Saved the data
        out_path = save_to_json(spinal_sections.tolist(), filepath.parent, "sections_sc.json")
        print(f"Saved section data from {mouse} into {out_path}")
        out_path = save_to_json(section_centers.tolist(), filepath.parent, "section_centers_sc.json")
        print(f"Saved section centers from {mouse} into {out_path}")
        out_data = [section.tolist() for section in section_points]
        out_path = save_to_json(out_data, filepath.parent, "section_points_sc.json")
        print(f"Saved section point data from {mouse} into {out_path}")
