# ================================================================
# 0. Section: IMPORTS
# ================================================================
import SimpleITK as sitk

from dataclasses import dataclass
from typing import ClassVar

from ...configs import MetricConfig



# ================================================================
# 1. Section: Functions
# ================================================================
@dataclass
class MetricFactory:
    loss_names: ClassVar[dict[str, list[str]]] = {
        "Mutual Information": ["MI", "Mutual Information", "Mattes Mutual Information"],
        "Mean Squares": ["LS", "Least Squares", "MSE", "Mean Squares"],
    }

    def apply(
        self,
        method: sitk.ImageRegistrationMethod,
        metric_config: MetricConfig,
    ) -> None:
        metric_name = metric_config.name.lower().strip()
        method.SetMetricSamplingPercentage(metric_config.sampling_percentage)

        if self._matches(metric_name, "Mutual Information"):
            method.SetMetricAsMattesMutualInformation(
                numberOfHistogramBins=metric_config.histogram_bins
            )

        elif self._matches(metric_name, "Mean Squares"):
            method.SetMetricAsMeanSquares()

        else:
            raise ValueError(
                f"The '{metric_config.name}' loss is not implemented. "
                f"Please select one of: {self.loss_names.keys()}"
            )


    # ──────────────────────────────────────────────────────
    # 1.1 Subsection: Helper Functions
    # ──────────────────────────────────────────────────────
    def _matches(self, metric_name: str, canonical_name: str) -> bool:
        aliases = self.loss_names[canonical_name]
        return metric_name in [alias.lower().strip() for alias in aliases]
