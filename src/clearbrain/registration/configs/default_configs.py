# ================================================================
# 0. Section: IMPORTS
# ================================================================
from .RegistrationConfig import RegistrationConfig
from .config import (
    MetricConfig,
    OptimizerConfig,
    InterpolationConfig,
    MultipleResolutionConfig,
)

# ================================================================
# 1. Section: Untwisting the Spinal Coord
# ================================================================
UNTWISTING_REGISTRATION = RegistrationConfig(
    transform_center=1,
    metric=MetricConfig(name="CC", sampling_percentage=1, histogram_bins=32),
    optimizer=OptimizerConfig(
        name="GD",
        iterations=5000,
        learning_rate=0.01,
        convergence_minimum_value=1e-5,
        convergence_window_size=10,
        gradient_convergence_tolerance=1e-8,
        maximum_number_corrections=5,
        estimate_learning_rate=2,
        max_step_size_physical_units=1.0,
        grid_size=2,
        constrains=[],
        scales=[],
        initial_angle=None,
    ),
    interpolator=InterpolationConfig(
        registration="bspline",
        resampling="linear",
    ),
    multiple_res=MultipleResolutionConfig(
        enabled=False,
    ),
)


# ================================================================
# 0. Section: IMPORTS
# ================================================================
TEMPLATE_WARP_REGISTRATION = RegistrationConfig(
    transform_center=1,
    metric=MetricConfig(name="MI", sampling_percentage=1, histogram_bins=32),
    optimizer=OptimizerConfig(
        name="GD",
        iterations=5000,
        learning_rate=0.01,
        convergence_minimum_value=1e-5,
        convergence_window_size=10,
        gradient_convergence_tolerance=1e-8,
        maximum_number_corrections=5,
        estimate_learning_rate=2,
        max_step_size_physical_units=1.0,
        grid_size=2,
        constrains=[],
        scales=[],
        initial_angle=None,
    ),
    interpolator=InterpolationConfig(
        registration="bspline",
        resampling="nearest",
    ),
    multiple_res=MultipleResolutionConfig(
        enabled=True,
        shrink_factors=[4, 2, 1],
        smoothing_sigmas=[2, 1, 0],
    ),
)
