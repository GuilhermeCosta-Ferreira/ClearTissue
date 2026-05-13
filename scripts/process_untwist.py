# ================================================================
# 0. Section: IMPORTS
# ================================================================
from matplotlib import pyplot as plt

from pathlib import Path

from clearbrain.processing import clear_external_points, rotate_spinal_cord, crop_excess
from clearbrain.tissue import TissueType
from clearbrain.data import TissueLoader, TissueSource, TissueDownloader
from clearbrain.tissue.view import plot_volume_coronal

# ================================================================
# 1. Section: INPUTS
# ================================================================
DATA_FOLDER: Path = Path("data")
MOUSE: str = "32B"
TISSUE_TYPE: TissueType = TissueType.SPINAL_COORD
SAMPLE_FACTOR: int = 25


# ================================================================
# 3. Section: MAIN
# ================================================================
if __name__ == "__main__":
    # 1. Load the untwisted tissue
    source = TissueSource(mouse=MOUSE, tissue_type=TISSUE_TYPE, base_path=DATA_FOLDER)
    loader = TissueLoader(source)
    tissue = loader.load_volume(suffix="_untwisted", sample_factor=SAMPLE_FACTOR)

    rotated_tissue = rotate_spinal_cord(tissue)
    plot_volume_coronal(rotated_tissue, 20)
    plt.show()

    cleaned_tissue = clear_external_points(rotated_tissue)
    plot_volume_coronal(cleaned_tissue, 20)
    plt.show()

    processed_untwist_volume = crop_excess(cleaned_tissue)
    plot_volume_coronal(processed_untwist_volume, 20)
    plt.show()

    downloader = TissueDownloader(source)
    p = downloader.download_volume(
        processed_untwist_volume, suffix="_untwisted_cleaned", to_update=True
    )
    print(f"Downloaded at {p}")
