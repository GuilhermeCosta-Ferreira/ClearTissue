# ================================================================
# 0. Section: IMPORTS
# ================================================================
import SimpleITK as sitk
import numpy as np

from dataclasses import dataclass


# ================================================================
# 1. Section: Functions
# ================================================================
@dataclass
class RegistrationResult:
    registered_image: np.ndarray
    transform: sitk.Transform
    final_metric: float
    stop_condition: str
    elapsed_time: float
