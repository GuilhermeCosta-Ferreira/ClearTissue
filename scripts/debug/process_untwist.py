# ================================================================
# 0. Section: IMPORTS
# ================================================================
from matplotlib import pyplot as plt

from pathlib import Path

from cleartissue.processing import clear_external_points, rotate_spinal_cord, crop_excess, apply_crop_excess, crop_spinal_cord
from cleartissue.tissue import TissueType
from cleartissue.data import TissueLoader, TissueSource, TissueDownloader
from cleartissue.tissue.view import plot_volume_coronal, plot_volume_overview



# ================================================================
# 1. Section: INPUTS
# ================================================================
DATA_FOLDER: Path = Path("data")
MOUSE: str = "32B"
TISSUE_TYPE: TissueType = TissueType.SPINAL_COORD
SAMPLE_FACTOR: int = 1


# ================================================================
# 3. Section: MAIN
# ================================================================
if __name__ == "__main__":
    # 1. Load the untwisted tissue
    source = TissueSource(mouse=MOUSE, tissue_type=TISSUE_TYPE, base_path=DATA_FOLDER)
    loader = TissueLoader(source)
    tissue = loader.load_volume(suffix="_tissue_untwisted", sample_factor=SAMPLE_FACTOR)
    cells = loader.load_volume(suffix="_cells_untwisted", sample_factor=SAMPLE_FACTOR)

    # 2. Rotate the tissue and cells
    pre_count = cells.volume.sum()
    rotated_tissue, angle = rotate_spinal_cord(tissue)
    rotated_cells, _ = rotate_spinal_cord(cells, angle)
    plot_volume_coronal(rotated_tissue, 20)
    plot_volume_coronal(rotated_cells, 20)
    plt.show()

    cleaned_tissue, margin = clear_external_points(rotated_tissue)
    cleaned_cells, _ = clear_external_points(rotated_cells, margin)
    plot_volume_coronal(cleaned_tissue, 20)
    plot_volume_coronal(cleaned_cells, 20)
    plt.show()

    processed_untwist_volume, crop_params = crop_excess(cleaned_tissue)
    processed_untwist_cells = apply_crop_excess(cleaned_cells, crop_params)
    plot_volume_coronal(processed_untwist_volume, 20)
    plot_volume_coronal(processed_untwist_cells, 20)
    plt.show()

    processed_cut_volume, start_slice, end_slice = crop_spinal_cord(processed_untwist_volume)
    processed_cut_cells, _, _ = crop_spinal_cord(processed_untwist_cells, start_slice, end_slice)
    plot_volume_overview(processed_cut_volume, 3)
    plot_volume_overview(processed_cut_cells, 3)
    plt.show()

    downloader = TissueDownloader(source)
    p = downloader.download_volume(
        processed_cut_volume, suffix="_tissue_untwisted_cleaned", to_update=True
    )
    print(f"Downloaded at {p}")
    downloader = TissueDownloader(source)
    p = downloader.download_volume(
        processed_cut_cells, suffix="_cells_untwisted_cleaned", to_update=True
    )
    print(f"Downloaded at {p}")
