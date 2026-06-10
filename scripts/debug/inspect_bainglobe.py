# ================================================================
# 0. Section: IMPORTS
# ================================================================
from pathlib import Path

import napari
import numpy as np
import tifffile



# ================================================================
# 1. Paths
# ================================================================
SAMPLE_NPY = Path("data/32B/tissue_sc_tissue_untwisted_cleaned_SF1.npy")
BRAINREG_OUTPUT = Path("work/brainreg_sample_reference")

REGISTERED_ATLAS = BRAINREG_OUTPUT / "registered_atlas.tiff"



# ================================================================
# 2. Load
# ================================================================
sample = np.load(SAMPLE_NPY)
registered_atlas = tifffile.imread(REGISTERED_ATLAS)


print("Sample shape:", sample.shape)
print("Registered atlas shape:", registered_atlas.shape)


# ================================================================
# 3. Display
# ================================================================
viewer = napari.Viewer()

viewer.add_image(
    sample,
    name="sample",
    contrast_limits=(np.percentile(sample, 1), np.percentile(sample, 99)),
)

viewer.add_labels(
    registered_atlas.astype(np.int32),
    name="registered_atlas",
    opacity=0.35,
)

napari.run()
