# ================================================================
# 0. Section: IMPORTS
# ================================================================
import numpy as np

from tqdm import tqdm
from typing import cast
from copy import deepcopy

from ...data import ClearVolume
from ....registration import Registrator, RegistrationResult



# ================================================================
# 1. Section: Functions
# ================================================================
def untwist_spinal_coord(
    tissue: ClearVolume, registrator: Registrator, window_size: int = 75, gap: int = 0
) -> tuple[ClearVolume, list]:
    # 0. Get the needed data
    volume = tissue.data
    nr_slices = volume.shape[0]

    # 1. Initilaize the required variables
    untwisted_volume = volume.astype(np.float32).copy()
    twisting_data = []

    previous_parameters = None
    previous_fixed_parameters = None

    # 2. Loop over every slice
    for sl in tqdm(range(nr_slices)):
        # 2.1. Skip the first slice
        if sl == 0:
            continue

        # 2.2. Start from the last rotation
        registrator.config.optimizer.initial_parameters = previous_parameters
        registrator.config.optimizer.initial_fixed_parameters = previous_fixed_parameters

        # 2.3. Get the moving
        moving = volume[sl, :, :].copy()

        # 2.4. get the reference slice as an average for robustness
        fixed = get_reference_slices(untwisted_volume, sl, window_size, gap)
        if fixed is None or np.sum(moving) <= 10:
            continue

        # 2.5. Do the registration and save it
        result = registrator.register(fixed, moving)
        twisting_data.append(result)
        untwisted_volume[sl, :, :] = deepcopy(result.registered_image)

        # 2.6. Update the previous angle
        #previous_angle = result.transform.GetParameters()[0]
        previous_parameters = tuple(result.transform.GetParameters())
        previous_fixed_parameters = tuple(result.transform.GetFixedParameters())


    # 3. Saves the untwisted as a volume
    tissue = tissue.copy_with(data=untwisted_volume)

    return tissue, twisting_data

def apply_know_untwisting(
    cells: ClearVolume, registrator: Registrator, registrator_result: list[RegistrationResult]
) -> ClearVolume:
    volume = cells.data
    nr_slices = volume.shape[0]

    untwisted_volume = volume.astype(np.float32).copy()

    for i in range(nr_slices):
        if i == 0:
            continue
        transform = registrator_result[i-1].transform
        untwisted_slice = cast(np.ndarray, registrator.apply(volume[i, :, :], untwisted_volume[i, :, :], transform, as_array=True))
        untwisted_volume[i, :, :] = untwisted_slice

    cells = cells.copy_with(data=untwisted_volume)

    return cells


# ──────────────────────────────────────────────────────
# 1.1 Subsection: Helper Functions
# ──────────────────────────────────────────────────────
def get_reference_slices(
    untwisted_volume: np.ndarray, current_slice: int, window_size: int, gap: int
) -> np.ndarray | None:
    end = max(1, current_slice - gap)
    start = max(0, end - window_size)
    previous_slices = untwisted_volume[start:end, :, :].astype(np.float32)

    # Ignore nearly empty reference slices if needed
    valid_refs = []
    for k in range(previous_slices.shape[0]):
        ref = previous_slices[k, :, :]
        if np.sum(ref) > 10:
            valid_refs.append(ref)

    if len(valid_refs) == 0:
        return None

    if len(valid_refs) == 1:
        return valid_refs[0].astype(np.float32)

    return np.mean(valid_refs, axis=0).astype(np.float32)
