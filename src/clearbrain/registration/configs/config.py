# ================================================================
# 0. Section: IMPORTS
# ================================================================
from dataclasses import dataclass, field



# ================================================================
# 1. Section: Functions
# ================================================================
@dataclass
class MetricConfig:
    name: str = "LS"
    sampling_percentage: float = 0.5
    histogram_bins: int = 32

@dataclass
class OptimizerConfig:
    name: str = "GD"
    iterations: int = 5000
    learning_rate: float = 0.1
    convergence_minimum_value: float = 1e-8
    convergence_window_size: int = 10
    gradient_convergence_tolerance: float = 1e-8
    maximum_number_corrections: int = 5
    estimate_learning_rate: int = 2 # 0, 1 or 2
    max_step_size_physical_units: float = 1.0
    grid_size: int = 2

@dataclass
class InterpolationConfig:
    registration: str = "linear"
    resampling: str = "linear"

@dataclass
class MultipleResolutionConfig:
    enabled: bool = False
    shrink_factors: list[int] = field(default_factory=lambda: [4, 2, 1])
    smoothing_sigmas: list[float] = field(default_factory=lambda: [2, 1, 0])
