# ================================================================
# 0. Section: IMPORTS
# ================================================================
from matplotlib import pyplot as plt

from pathlib import Path

from clearbrain.processing import clear_external_points
from clearbrain.tissue import TissueType
from clearbrain.data import TissueLoader, TissueSource
from clearbrain.tissue.view import plot_volume_coronal



# ================================================================
# 1. Section: INPUTS
# ================================================================
DATA_FOLDER: Path = Path("data")
MOUSE: str = "32B"
TISSUE_TYPE: TissueType = TissueType.SPINAL_COORD
SAMPLE_FACTOR: int = 25




# ================================================================
# 2. Section: FUNCTIONS
# ================================================================



# ================================================================
# 3. Section: MAIN
# ================================================================
if __name__ == '__main__':
    # 1. Load the untwisted tissue
    source = TissueSource(
        mouse = MOUSE,
        tissue_type = TISSUE_TYPE,
        base_path = DATA_FOLDER
    )
    loader = TissueLoader(source)
    tissue = loader.load_volume(suffix="_untwisted", sample_factor=SAMPLE_FACTOR)

    cleaned_tissue = clear_external_points(tissue)
    plot_volume_coronal(cleaned_tissue, 20)
    plt.show()
