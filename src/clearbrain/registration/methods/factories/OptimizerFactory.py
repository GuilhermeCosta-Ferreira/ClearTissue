# ================================================================
# 0. Section: IMPORTS
# ================================================================
import SimpleITK as sitk

from dataclasses import dataclass
from typing import ClassVar

from ...configs import OptimizerConfig



# ================================================================
# 1. Section: Functions
# ================================================================
@dataclass
class OptimizerFactory:
    optimizer_names: ClassVar[dict[str, list[str]]] = {
        "Gradient Descent": ["GD", "Gradient Descent"],
        "LBFGS": ["LBFGS"],
    }

    sitk.ImageRegistrationMethod.Once

    def apply(
        self,
        method: sitk.ImageRegistrationMethod,
        optimizer_config: OptimizerConfig,
    ) -> None:
        optimizer_name = optimizer_config.name.lower().strip()

        if self._matches(optimizer_name, "LBFGS"):
            method.SetOptimizerAsLBFGSB(
                gradientConvergenceTolerance=optimizer_config.gradient_convergence_tolerance,
                numberOfIterations=optimizer_config.iterations,
                maximumNumberOfCorrections=optimizer_config.maximum_number_corrections
            )
        elif self._matches(optimizer_name, "Gradient Descent"):
            method.SetOptimizerAsGradientDescent(
                learningRate=optimizer_config.learning_rate,
                numberOfIterations=optimizer_config.iterations,
                convergenceMinimumValue=optimizer_config.convergence_minimum_value,
                convergenceWindowSize=optimizer_config.convergence_window_size,
                estimateLearningRate=optimizer_config.estimate_learning_rate,
                maximumStepSizeInPhysicalUnits=optimizer_config.max_step_size_physical_units
            )
        else:
            raise ValueError(
                f"The '{optimizer_config.name}' loss is not implemented. "
                f"Please select one of: {self.optimizer_names.keys()}"
            )


    # ──────────────────────────────────────────────────────
    # 1.1 Subsection: Helper Functions
    # ──────────────────────────────────────────────────────
    def _matches(self, metric_name: str, canonical_name: str) -> bool:
        aliases = self.optimizer_names[canonical_name]
        return metric_name in [alias.lower().strip() for alias in aliases]
