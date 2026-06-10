# ================================================================
# 0. Section: IMPORTS
# ================================================================
import numpy as np
from matplotlib import pyplot as plt

from pathlib import Path

from clearbrain.tissue import TissueType, ClearVolume
from clearbrain.data import TissueLoader, TissueSource, TissueDownloader



# ================================================================
# 1. Section: INPUTS
# ================================================================
DATA_FOLDER: Path = Path("data")
MOUSE: str = "32B"
TISSUE_TYPE: TissueType = TissueType.SPINAL_COORD



# ================================================================
# 2. Section: FUNCTIONS
# ================================================================



# ================================================================
# 3. Section: MAIN
# ================================================================
if __name__ == '__main__':
    source = TissueSource(mouse=MOUSE, tissue_type=TISSUE_TYPE, base_path=DATA_FOLDER)
    loader = TissueLoader(source)
    tissue = loader.load_volume(suffix="_tissue_untwisted_cleaned")
    reg_atlas = loader.load_volume(suffix="_registered_atlas")

    reg_centers = []

    for i in range(reg_atlas.volume.shape[1]):
        reg_atlas_slice = reg_atlas.volume[:, i, :]

        coords = np.argwhere(reg_atlas_slice > 0)

        if coords.size == 0:
            reg_centers.append([np.nan, np.nan])
            continue

        reg_atlas_center = coords.mean(axis=0)
        reg_centers.append(reg_atlas_center)

    reg_centers = np.asarray(reg_centers)

    grad = np.gradient(reg_centers, axis=0)
    magnitude = np.linalg.norm(grad, axis=1)
    std = np.std(grad)

    plt.figure()
    plt.plot(magnitude)
    plt.plot([std*3] * len(magnitude))
    plt.xlabel("Slice index")
    plt.ylabel("Centre displacement per slice")
    plt.title("Registered atlas centre movement")
    plt.show()

    valid_frames = np.asarray([True] * len(magnitude))

    counted = 0
    empty_slices = 0
    for idx, fr in enumerate(valid_frames):
        if counted % 2 == 1:
            if magnitude[idx] > std * 3:
                valid_frames[idx] = False
                counted += 1
                empty_slices = 0
            elif counted > 0 and empty_slices < 3:
                valid_frames[idx] = False
                empty_slices += 1
            else:
                empty_slices += 1
                counted += 1
        else:
            if magnitude[idx] > std * 3:
                valid_frames[idx] = False
                counted += 1
                empty_slices = 0
            else:
                empty_slices += 1



    plt.figure()
    plt.plot(magnitude)
    plt.plot([std*3] * len(magnitude))
    plt.scatter(np.arange(len(magnitude))[~valid_frames], magnitude[~valid_frames], color='red')
    plt.xlabel("Slice index")
    plt.ylabel("Centre displacement per slice")
    plt.title("Registered atlas centre movement")
    plt.show()

    atlast_reg_data = loader.load_twisting_data(suffix="_atlas_registration_data")

    warp_regs = [reg[1] for reg in atlast_reg_data]
    final_metrics = [res.final_metric for res in warp_regs]

    plt.figure()
    plt.plot(final_metrics)
    plt.xlabel("Slice index")
    plt.ylabel("Registration metric")
    plt.title("Registration metrics over slices")
    plt.show()
