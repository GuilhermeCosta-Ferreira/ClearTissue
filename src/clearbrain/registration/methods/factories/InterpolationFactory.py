# ================================================================
# 0. Section: IMPORTS
# ================================================================
import SimpleITK as sitk

from dataclasses import dataclass
from typing import ClassVar

from ...configs import InterpolationConfig



# ================================================================
# 1. Section: Functions
# ================================================================
@dataclass
class InterpolationFactory:
    interpolation_names: ClassVar[dict[str, list[str]]] = {
        "Linear": ["Linear"],
        "Nearest": ["NN", "Nearest", "Nearest Neighbour"],
    }

    def apply_registration(
        self,
        method: sitk.ImageRegistrationMethod,
        interpolation_config: InterpolationConfig,
    ) -> None:
        interpolation_names = interpolation_config.registration.lower().strip()

        if self._matches(interpolation_names, "Linear"):
            method.SetInterpolator(sitk.sitkLinear)
        elif self._matches(interpolation_names, "Nearest"):
            method.SetInterpolator(sitk.sitkNearestNeighbor)
        else:
            raise ValueError(
                f"The '{interpolation_config.registration}' interpolation is not "
                f"implemented. Please select one of: {self.interpolation_names.keys()}"
            )

    def apply_resampling(
        self,
        method: sitk.ResampleImageFilter,
        interpolation_config: InterpolationConfig,
    ) -> None:
        interpolation_names = interpolation_config.resampling.lower().strip()

        if self._matches(interpolation_names, "Linear"):
            method.SetInterpolator(sitk.sitkLinear)
        elif self._matches(interpolation_names, "Nearest"):
            method.SetInterpolator(sitk.sitkNearestNeighbor)
        else:
            raise ValueError(
                f"The '{interpolation_config.resampling}' interpolation is not "
                f"implemented. Please select one of: {self.interpolation_names.keys()}"
            )


    # ──────────────────────────────────────────────────────
    # 1.1 Subsection: Helper Functions
    # ──────────────────────────────────────────────────────
    def _matches(self, metric_name: str, canonical_name: str) -> bool:
        aliases = self.interpolation_names[canonical_name]
        return metric_name in [alias.lower().strip() for alias in aliases]
