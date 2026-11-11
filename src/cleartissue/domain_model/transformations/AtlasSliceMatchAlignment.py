# ================================================================
# 0. Section: IMPORTS
# ================================================================
import numpy as np

from numpy.typing import NDArray
from dataclasses import dataclass

from ...registration.strategies import AffineRegistration
from .AbstractProfileAlignment import AbstractProfileAlignment
from ...registration.QuickRegistrator import QuickRegistrator
from ...registration import RegistrationConfig, RegistratorResampler
from ..analysis.cord_profile import unique_slice_runs, vote_shift


# ================================================================
# 1. Section: Functions
# ================================================================
@dataclass
class AtlasSliceMatchAlignment(AbstractProfileAlignment):
    """Align by registering every unique atlas slice to each head tissue slice.

    Each tissue slice in the first head_fraction of the cord gets a 2D affine
    registration from every unique atlas slice (masks); candidates are ranked
    by final registration loss (lower is better) and the top_k vote for the
    shift with weights k..1 (vote_shift, Gaussian-smoothed by vote_sigma).
    Costs n_head x n_unique registrations, so keep the registrator light.
    The atlas is cropped / zero-padded along axis 0 so tissue slice i lines up
    with atlas slice i (see AbstractProfileAlignment).

    Attributes:
        atlas_affine_registrator_params: RegistrationConfig dict for the 2D affine.
        head_fraction: Fraction of usable tissue slices, from the head, that vote.
        top_k: Matches per tissue slice that vote.
        vote_sigma: Gaussian smoothing of the vote histogram, in slices.
    """

    atlas_affine_registrator_params: dict
    head_fraction: float = 0.3
    top_k: int = 3
    vote_sigma: float = 2.0

    def __post_init__(self):
        # Quick variant: only the final metric is needed, never the resampled slice.
        self.registrator = QuickRegistrator(
            strategy=AffineRegistration(),
            resampler=RegistratorResampler(),
            config=RegistrationConfig.from_dict(self.atlas_affine_registrator_params),
        )

    def find_shift(
        self, tissue: NDArray, atlas: NDArray, excluded: NDArray | None
    ) -> int:
        tissue_mask, atlas_mask = tissue > 0, atlas > 0
        head_idx = _head_slices(tissue_mask, excluded, self.head_fraction)
        reps = np.array([(s + e - 1) // 2 for s, e in unique_slice_runs(atlas_mask)])

        losses = np.array(
            [
                [
                    self.registrator.register(
                        tissue_mask[t], atlas_mask[a]
                    ).final_metric
                    for a in reps
                ]
                for t in head_idx
            ]
        )
        order = np.argsort(losses, axis=1)[:, : min(self.top_k, len(reps))]
        # Rank-only weights: raw losses aren't comparable across tissue slices.
        shifts, votes = vote_shift(
            head_idx,
            reps[order],
            np.ones(order.shape),
            len(atlas),
            len(tissue),
            self.vote_sigma,
        )
        return int(shifts[np.argmax(votes)])


# ──────────────────────────────────────────────────────
# 1.1 Subsection: Helper Functions
# ──────────────────────────────────────────────────────
def _head_slices(
    tissue_mask: NDArray, excluded: NDArray | None, head_fraction: float
) -> NDArray:
    usable = tissue_mask.reshape(len(tissue_mask), -1).any(axis=1)
    if excluded is not None:
        usable &= ~excluded
    filled = np.flatnonzero(usable)
    if len(filled) == 0:
        raise ValueError("No usable tissue slices; nothing to align.")
    return filled[: max(1, int(round(len(filled) * head_fraction)))]
