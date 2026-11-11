# ================================================================
# 0. Section: IMPORTS
# ================================================================
from numpy.typing import NDArray
from dataclasses import dataclass, field

from .CorrelationAlignment import CorrelationAlignment
from .AtlasSliceMatchAlignment import AtlasSliceMatchAlignment
from .AbstractProfileAlignment import AbstractProfileAlignment
from .utils.alignment import ratio_profile
from ..analysis.cord_profile import fit_ratio_bump, pca_elongation_profiles

_INNER_ALIGNMENTS: dict[str, type[AbstractProfileAlignment]] = {
    "CorrelationAlignment": CorrelationAlignment,
    "AtlasSliceMatchAlignment": AtlasSliceMatchAlignment,
}


# ================================================================
# 1. Section: Functions
# ================================================================
@dataclass
class BumpExcludedAlignment(AbstractProfileAlignment):
    """Run another alignment while ignoring the cord-manipulation bump.

    The manipulation shows up as a Gaussian bump in the per-slice PC1/PC2
    ratio; slices within centre ± exclude_sigmas * sigma are masked out and
    the inner alignment scores only the rest. If the bump fit fails, the inner
    alignment runs unmasked. The atlas is cropped / zero-padded along axis 0
    so tissue slice i lines up with atlas slice i (see AbstractProfileAlignment).

    Attributes:
        inner: Class name of the wrapped alignment (a key of _INNER_ALIGNMENTS).
        inner_params: Keyword arguments for the wrapped alignment.
        exclude_sigmas: Half-width of the masked stretch, in bump sigmas.
        smooth_slices: Median window applied to the ratio before fitting.
    """

    inner: str = "CorrelationAlignment"
    inner_params: dict = field(default_factory=dict)
    exclude_sigmas: float = 1.5
    smooth_slices: int = 15

    def __post_init__(self):
        self.inner_alignment = _INNER_ALIGNMENTS[self.inner](**self.inner_params)

    def find_shift(
        self, tissue: NDArray, atlas: NDArray, excluded: NDArray | None
    ) -> int:
        pca = pca_elongation_profiles(tissue)
        bump = fit_ratio_bump(ratio_profile(pca.pc1, pca.pc2), self.smooth_slices)
        if bump is None:
            print(
                f"{type(self).__name__}: bump fit failed, running {self.inner} unmasked"
            )
            return self.inner_alignment.find_shift(tissue, atlas, excluded)

        bump_mask = bump.excluded(self.exclude_sigmas)
        if excluded is not None:
            bump_mask |= excluded
        return self.inner_alignment.find_shift(tissue, atlas, bump_mask)
