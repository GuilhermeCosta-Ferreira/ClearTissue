# ================================================================
# 0. Section: IMPORTS
# ================================================================
import numpy as np

from tqdm import tqdm

from ..tissue import ClearVolume, SpinalCenterline
from ..registration import RegistrationConfig, Registrator, RegistratorResampler, RigidRegistration


# ================================================================
# 1. Section: Functions
# ================================================================
def untwist_spinal_coord(tissue_volume: ClearVolume, centerline: SpinalCenterline) -> ClearVolume:
    volume = tissue_volume.volume
    nr_slices = volume.shape[1]

    registrator = Registrator(
        strategy = RigidRegistration(),
        resampler = RegistratorResampler(),
        config = RegistrationConfig()
    )

    untwisted_volume = volume.copy()
    twisting_data = []
    for sl in tqdm(range(nr_slices)):
        if sl == 0:
            continue

        fixed = untwisted_volume[:, sl-1, :]
        moving = volume[:, sl, :]

        if np.sum(fixed) <= 50 or np.sum(moving) <= 50:
            continue

        result = registrator.register(fixed, moving)
        twisting_data.append(result)

        untwisted_volume[:, sl, :] = result.registered_image

    return ClearVolume(untwisted_volume, tissue_volume.metadata, tissue_volume.sample_factor)




# ──────────────────────────────────────────────────────
# 1.1 Subsection: Helper Functions
# ──────────────────────────────────────────────────────
"""
fig, axes = plt.subplots(2,2)
imgs = [fixed, moving, fixed, result.registered_image]
titles = ["fixed", "moving", "fixed", "registered_image"]
axes = axes.ravel()

for idx, ax in enumerate(axes):
    ax.imshow(imgs[idx], cmap="hot")
    ax.set_title(titles[idx])

plt.tight_layout()
plt.show()"""
