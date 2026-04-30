# ================================================================
# 0. Section: IMPORTS
# ================================================================
from dataclasses import dataclass



# ================================================================
# 1. Section: Functions
# ================================================================
@dataclass
class MetricConfig:
    name: str
    sampling_percentage: float
    histogram_bins: int

@dataclass
class OptimizerConfig:
    name: str
    iterations: int
    learning_rate: float
    convergence_minimum_value: float
    convergence_window_size: int
    gradient_convergence_tolerance: float
    maximum_number_corrections: int
    estimate_learning_rate: int # 0, 1 or 2
    max_step_size_physical_units: float
    grid_size: int

@dataclass
class InterpolationConfig:
    registration: str
    resampling: str

@dataclass
class MultipleResolutionConfig:
    enabled: bool
    shrink_factors: list[int]
    smoothing_sigmas: list[float]
