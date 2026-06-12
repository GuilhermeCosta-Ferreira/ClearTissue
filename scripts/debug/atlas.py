# ================================================================
# 0. Section: IMPORTS
# ================================================================
import numpy as np
from matplotlib import pyplot as plt

from pathlib import Path
from rich.pretty import pprint
from scipy.ndimage import zoom
from brainglobe_atlasapi import BrainGlobeAtlas

from clearbrain.data import Metadata
from clearbrain.tissue import ClearVolume, TissueType
from clearbrain.tissue.view import plot_volume_overview


# ================================================================
# 1. Section: INPUTS
# ================================================================



# ================================================================
# 2. Section: FUNCTIONS
# ================================================================
def resample_to_isotropic(
    volume: np.ndarray,
    old_resolution_um: tuple[float, float, float],
    target_resolution_um: float,
    is_label: bool = False,
) -> np.ndarray:
    old_resolution_um = np.asarray(old_resolution_um, dtype=float)

    zoom_factors = old_resolution_um / target_resolution_um

    order = 0 if is_label else 1

    resampled = zoom(
        volume,
        zoom=zoom_factors,
        order=order,
    )

    if is_label:
        resampled = resampled.astype(volume.dtype)

    return resampled


# ================================================================
# 3. Section: MAIN
# ================================================================
if __name__ == '__main__':
    atlas = BrainGlobeAtlas("allen_cord_20um", check_latest=False)

    pprint(atlas.orientation)
    pprint(atlas.resolution)
    pprint(atlas.shape)
    pprint(atlas.shape_um)
    pprint(atlas.metadata)
    pprint(atlas.space)

    old_resolution = tuple(atlas.resolution)  # (20.0, 10.0, 10.0)

    annotation_iso = resample_to_isotropic(
        atlas.annotation,
        old_resolution_um=old_resolution,
        target_resolution_um=25.0,
        is_label=True,
    )

    print(annotation_iso.shape)

    plot_volume_overview(ClearVolume(volume=annotation_iso, metadata=Metadata(mouse="", tissue_type=TissueType.SPINAL_COORD, file_path=Path("")), sample_factor=1), nr_cuts=3)
    plt.show()
