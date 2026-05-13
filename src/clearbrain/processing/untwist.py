# ================================================================
# 0. Section: IMPORTS
# ================================================================
import numpy as np

from tqdm import tqdm

from ..tissue import ClearVolume
from ..registration import Registrator


# ================================================================
# 1. Section: Functions
# ================================================================
def untwist_spinal_coord(
    tissue_volume: ClearVolume, registrator: Registrator, window_size: int = 75
) -> tuple[ClearVolume, list]:
    # 0. Get the needed data
    volume = tissue_volume.volume
    nr_slices = volume.shape[1]

    # 1. Initilaize the required variables
    untwisted_volume = volume.astype(np.float32).copy()
    twisting_data = []
    previous_angle = 0

    # 2. Loop over every slice
    for sl in tqdm(range(nr_slices)):
        # 2.1. Skip the first slice
        if sl == 0:
            continue

        # 2.2. Start from the last rotation
        registrator.config.optimizer.initial_angle = previous_angle

        # 2.3. Get the moving
        moving = volume[:, sl, :].copy()

        # 2.4. get the reference slice as an average for robustness
        fixed = get_reference_slices(untwisted_volume, sl, window_size)
        if fixed is None or np.sum(moving) <= 10:
            continue

        # 2.5. Do the registration and save it
        result = registrator.register(fixed, moving)
        twisting_data.append(result)
        untwisted_volume[:, sl, :] = result.registered_image

        # 2.6. Update the previous angle
        previous_angle = result.transform.GetParameters()[0]

    # 3. Saves the untwisted as a volume
    tissue_volume = ClearVolume(
        untwisted_volume, tissue_volume.metadata, tissue_volume.sample_factor
    )

    return tissue_volume, twisting_data


# ──────────────────────────────────────────────────────
# 1.1 Subsection: Helper Functions
# ──────────────────────────────────────────────────────
def get_reference_slices(
    untwisted_volume: np.ndarray, current_slice: int, window_size: int
) -> np.ndarray | None:
    start = max(0, current_slice - window_size)
    previous_slices = untwisted_volume[:, start:current_slice, :].astype(np.float32)

    # Ignore nearly empty reference slices if needed
    valid_refs = []
    for k in range(previous_slices.shape[1]):
        ref = previous_slices[:, k, :]
        if np.sum(ref) > 10:
            valid_refs.append(ref)

    if len(valid_refs) == 0:
        return None
    else:
        return np.mean(valid_refs, axis=0).astype(np.float32)


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

plt.figure()
angles = []
for result in twisting_data:
    a = result.transform.GetParameters()[0]
    angles.append(math.degrees(a))
plt.plot(angles)
plt.show(block=False)
plt.figure()

angles = []
for result in twisting_data:
    a = result.elapsed_time
    angles.append(a)
plt.plot(angles)
plt.title("elapsed time per frame")
plt.show(block=False)


#if (30 <= sl <= 50) or (330 <= sl <= 350):
#if (330 <= sl <= 340):
if False:
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

"""
