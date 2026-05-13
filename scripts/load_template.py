# ================================================================
# 0. Section: IMPORTS
# ================================================================
import numpy as np
from matplotlib import pyplot as plt

from pathlib import Path
from brainglobe_atlasapi import BrainGlobeAtlas

from clearbrain.tissue import TissueType
from clearbrain.data import TissueLoader, TissueSource
from clearbrain.registration import Registrator, RegistratorResampler
from clearbrain.registration.configs import TEMPLATE_WARP_REGISTRATION
from clearbrain.registration.strategies import BSplineRegistration

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
if __name__ == "__main__":
    atlas = BrainGlobeAtlas("allen_cord_20um", check_latest=False)

    tree = atlas.hierarchy
    reference = atlas.reference  # 3D anatomical image
    annotation = atlas.annotation  # 3D label image / region IDs
    metadata = atlas.metadata

    print(reference.shape)
    print(annotation.shape)
    print(metadata)
    print(metadata.keys())
    print(tree)

    plt.figure()
    plt.imshow(annotation[500, :, :])
    plt.show(block=False)

    source = TissueSource(mouse=MOUSE, tissue_type=TISSUE_TYPE, base_path=DATA_FOLDER)
    loader = TissueLoader(source)

    tissue = loader.load_volume(suffix="_untwisted", sample_factor=25)

    plt.figure()
    plt.imshow(tissue.volume[:, 250, :])
    plt.show(block=False)

    sample_tissue = np.where(tissue.volume[:, 250, :] > 0, 1, 0)
    sample_template = np.where(annotation[500, :, :] > 0, 1, 0)

    registrator = Registrator(
        strategy=BSplineRegistration(),
        resampler=RegistratorResampler(),
        config=TEMPLATE_WARP_REGISTRATION,
    )

    result = registrator.register(sample_tissue, sample_template)
    print(result.final_metric)
    print(result.stop_condition)

    plt.figure()
    plt.imshow(sample_tissue, alpha=0.5, cmap="Reds")
    plt.imshow(result.registered_image, alpha=0.5)
    plt.show()
