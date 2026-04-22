# ================================================================
# 0. Section: IMPORTS
# ================================================================
from clearbrain.sections.pca import get_image_pca_components
from matplotlib import pyplot as plt

from pathlib import Path

from clearbrain.sections.projection import get_2d_sections
from clearbrain.sections.plot_projections import plot_section_2d
from clearbrain import load_points



# ================================================================
# 1. Section: INPUTS
# ================================================================
# IO Settings
DATA_FOLDER: Path = Path("data")
MICE: list = ["32B"]
FILE_TARGET: str = "raw_points_sc.json"

DENSITY_RADIUS: int = 40

HIGHLIGHT_CENTERLINE: bool = True # makes sure the line is drawn on top of it
PLOT_SUBSAMPLE: int = 80  # Get's every X points



# ================================================================
# 3. Section: MAIN
# ================================================================
if __name__ == '__main__':
    for mouse in MICE:
        filepath = DATA_FOLDER / mouse / FILE_TARGET
        centerline_path = DATA_FOLDER / mouse / "centerline_sc.json"
        section_points_path = DATA_FOLDER / mouse / "section_points_sc.json"
        section_centers_path = DATA_FOLDER / mouse / "section_centers_sc.json"

        # 1. Load the points
        centerline = load_points(centerline_path)
        section_points = load_points(section_points_path)
        section_centers = load_points(section_centers_path)

        prism_imgs = get_2d_sections(section_points, section_centers, centerline)

        print(get_image_pca_components(prism_imgs[0]))

        plot_section_2d(prism_imgs)
        plt.show()
