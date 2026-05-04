# ================================================================
# 0. Section: IMPORTS
# ================================================================
import math

import numpy as np
from matplotlib import pyplot as plt

from tqdm import tqdm

from ..tissue import ClearVolume, SpinalCenterline
from ..registration import RegistrationConfig, Registrator, RegistratorResampler, RotationRigidRegistration


# ================================================================
# 1. Section: Functions
# ================================================================
def untwist_spinal_coord(tissue_volume: ClearVolume, centerline: SpinalCenterline) -> ClearVolume:
    volume = tissue_volume.volume
    nr_slices = volume.shape[1]

    angle_step = math.radians(0.01)
    max_angle = math.radians(2.5)
    n_steps = int(max_angle / angle_step)


    config = RegistrationConfig()
    #config.optimizer.name = "constrained"
    config.optimizer.constrains = [n_steps, 0, 0]
    config.optimizer.scales = [angle_step, 1.0, 1.0]

    registrator = Registrator(
        strategy = RotationRigidRegistration(),
        resampler = RegistratorResampler(),
        config = config
    )

    untwisted_volume = volume.astype(np.float32).copy()
    twisting_data = []
    previous_angle = 0
    for sl in tqdm(range(nr_slices)):
        if sl == 0:
            continue

        registrator.config.optimizer.initial_angle = np.max([previous_angle, 0.0])

        fixed = untwisted_volume[:, sl-1, :].copy()
        moving = volume[:, sl, :].copy()

        if np.sum(fixed) <= 10 or np.sum(moving) <= 10:
            continue

        result = registrator.register(fixed, moving)
        twisting_data.append(result)

        #if (30 <= sl <= 50) or (330 <= sl <= 350):
        if (330 <= sl <= 340):
            print("fixed sum:", np.sum(fixed))
            print("untwisted previous sum:", np.sum(untwisted_volume[:, sl-1, :]))
            print("moving sum:", np.sum(moving))

            print(volume.dtype)
            print(untwisted_volume.dtype)
            print(result.registered_image.dtype)

            fig, axes = plt.subplots(2,2)
            imgs = [fixed, moving, fixed, result.registered_image]
            titles = [f"fixed{np.sum(fixed)}", f"moving{np.sum(moving)}", f"fixed{np.sum(fixed) - np.sum(moving)}", f"registered_image{np.sum(result.registered_image)}"]
            axes = axes.ravel()

            for idx, ax in enumerate(axes):
                ax.imshow(imgs[idx], cmap="hot")
                ax.set_title(titles[idx])

            plt.tight_layout()
            plt.show(block=False)

            plt.figure()
            angles = []
            for result in twisting_data:
                a = result.transform.GetParameters()[0]
                angles.append(math.degrees(a))
            plt.plot(angles)
            plt.show()

        #print((result.registered_image == untwisted_volume[:, sl, :]).all())
        untwisted_volume[:, sl, :] = result.registered_image
        #print((result.registered_image == untwisted_volume[:, sl, :]).all())

    plt.figure()
    angles = []
    for result in twisting_data:
        a = result.transform.GetParameters()[0]
        angles.append(math.degrees(a))
    plt.plot(angles)
    plt.show(block=False)

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
plt.show()

if(sl % 10 == 0):
    plt.figure()
    angles = []
    for result in twisting_data:
        a = result.transform.GetParameters()[0]
        angles.append(math.degrees(a))
    plt.plot(angles)
    plt.show()
"""
