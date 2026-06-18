# ================================================================
# 0. Section: IMPORTS
# ================================================================
from typing import Self, cast, Any
from dataclasses import dataclass, field, replace

_UNSET = object()



# ================================================================
# 1. Section: Functions
# ================================================================
@dataclass
class MetricConfig:
    name: str = "CC"
    sampling_percentage: float = 1
    histogram_bins: int = 32

    @classmethod
    def from_dict(cls, data: dict) -> "MetricConfig":
        return cls(
            name=data.get("name", "CC"),
            sampling_percentage=data.get("sampling_percentage", 1),
            histogram_bins=data.get("histogram_bins", 32),
        )

    def copy_with(
        self,
        *,
        name: str | object = _UNSET,
        sampling_percentage: float | object = _UNSET,
        histogram_bins: int | object = _UNSET,
    ) -> Self:
        return replace(
            self,
            name=self.name if name is _UNSET else cast(str, name),
            sampling_percentage=(
                self.sampling_percentage
                if sampling_percentage is _UNSET
                else cast(float, sampling_percentage)
            ),
            histogram_bins=(
                self.histogram_bins
                if histogram_bins is _UNSET
                else cast(int, histogram_bins)
            ),
        )




@dataclass
class OptimizerConfig:  # pylint: disable=too-many-instance-attributes
    name: str = "GD"
    iterations: int = 5000
    learning_rate: float = 0.01
    convergence_minimum_value: float = 1e-5
    convergence_window_size: int = 10
    gradient_convergence_tolerance: float = 1e-8
    maximum_number_corrections: int = 5
    estimate_learning_rate: int = 2  # 0, 1 or 2
    max_step_size_physical_units: float = 1.0
    grid_size: int = 2
    constrains: list = field(default_factory=lambda: [])
    scales: list = field(default_factory=lambda: [])
    initial_angle: float | None = None
    initial_fixed_parameters: tuple[float, ...] | None = None
    initial_parameters: tuple[float, ...] | None = None

    @classmethod
    def from_dict(cls, data: dict) -> "OptimizerConfig":
        return cls(
            name=data.get("name", "GD"),
            iterations=data.get("iterations", 5000),
            learning_rate=data.get("learning_rate", 0.01),
            convergence_minimum_value=data.get("convergence_minimum_value", 1e-5),
            convergence_window_size=data.get("convergence_window_size", 10),
            gradient_convergence_tolerance=data.get("gradient_convergence_tolerance", 1e-8),
            maximum_number_corrections=data.get("maximum_number_corrections", 5),
            estimate_learning_rate=data.get("estimate_learning_rate", 2),
            max_step_size_physical_units=data.get("max_step_size_physical_units", 1.0),
            grid_size=data.get("grid_size", 2),
            constrains=data.get("constrains", []),
            scales=data.get("scales", []),
            initial_angle=data.get("initial_angle", None),
            initial_fixed_parameters=data.get("initial_fixed_parameters", None),
            initial_parameters=data.get("initial_parameters", None),
        )

    def copy_with(
        self,
        *,
        name: str | object = _UNSET,
        iterations: int | object = _UNSET,
        learning_rate: float | object = _UNSET,
        convergence_minimum_value: float | object = _UNSET,
        convergence_window_size: int | object = _UNSET,
        gradient_convergence_tolerance: float | object = _UNSET,
        maximum_number_corrections: int | object = _UNSET,
        estimate_learning_rate: int | object = _UNSET,
        max_step_size_physical_units: float | object = _UNSET,
        grid_size: int | object = _UNSET,
        constrains: list[Any] | object = _UNSET,
        scales: list[Any] | object = _UNSET,
        initial_angle: float | None | object = _UNSET,
        initial_fixed_parameters: tuple[float, ...] | None | object = _UNSET,
        initial_parameters: tuple[float, ...] | None | object = _UNSET,
    ) -> Self:
        return replace(
            self,
            name=self.name if name is _UNSET else cast(str, name),
            iterations=(
                self.iterations
                if iterations is _UNSET
                else cast(int, iterations)
            ),
            learning_rate=(
                self.learning_rate
                if learning_rate is _UNSET
                else cast(float, learning_rate)
            ),
            convergence_minimum_value=(
                self.convergence_minimum_value
                if convergence_minimum_value is _UNSET
                else cast(float, convergence_minimum_value)
            ),
            convergence_window_size=(
                self.convergence_window_size
                if convergence_window_size is _UNSET
                else cast(int, convergence_window_size)
            ),
            gradient_convergence_tolerance=(
                self.gradient_convergence_tolerance
                if gradient_convergence_tolerance is _UNSET
                else cast(float, gradient_convergence_tolerance)
            ),
            maximum_number_corrections=(
                self.maximum_number_corrections
                if maximum_number_corrections is _UNSET
                else cast(int, maximum_number_corrections)
            ),
            estimate_learning_rate=(
                self.estimate_learning_rate
                if estimate_learning_rate is _UNSET
                else cast(int, estimate_learning_rate)
            ),
            max_step_size_physical_units=(
                self.max_step_size_physical_units
                if max_step_size_physical_units is _UNSET
                else cast(float, max_step_size_physical_units)
            ),
            grid_size=(
                self.grid_size
                if grid_size is _UNSET
                else cast(int, grid_size)
            ),
            constrains=(
                list(self.constrains)
                if constrains is _UNSET
                else list(cast(list[Any], constrains))
            ),
            scales=(
                list(self.scales)
                if scales is _UNSET
                else list(cast(list[Any], scales))
            ),
            initial_angle=(
                self.initial_angle
                if initial_angle is _UNSET
                else cast(float | None, initial_angle)
            ),
            initial_fixed_parameters=(
                self.initial_fixed_parameters
                if initial_fixed_parameters is _UNSET
                else cast(tuple[float, ...] | None, initial_fixed_parameters)
            ),
            initial_parameters=(
                self.initial_parameters
                if initial_parameters is _UNSET
                else cast(tuple[float, ...] | None, initial_parameters)
            ),
        )



@dataclass
class InterpolationConfig:
    registration: str = "bspline"
    resampling: str = "linear"

    @classmethod
    def from_dict(cls, data: dict) -> "InterpolationConfig":
        return cls(
            registration=data.get("registration", "bspline"),
            resampling=data.get("resampling", "linear"),
        )

    def copy_with(
        self,
        *,
        registration: str | object = _UNSET,
        resampling: str | object = _UNSET,
    ) -> Self:
        return replace(
            self,
            registration=(
                self.registration
                if registration is _UNSET
                else registration
            ),
            resampling=(
                self.resampling
                if resampling is _UNSET
                else resampling
            ),
        )



@dataclass
class MultipleResolutionConfig:
    enabled: bool = False
    shrink_factors: list[int] = field(default_factory=lambda: [4, 2, 1])
    smoothing_sigmas: list[float] = field(default_factory=lambda: [2, 1, 0])

    @classmethod
    def from_dict(cls, data: dict) -> "MultipleResolutionConfig":
        return cls(
            enabled=data.get("enabled", False),
            shrink_factors=data.get("shrink_factors", [4, 2, 1]),
            smoothing_sigmas=data.get("smoothing_sigmas", [2, 1, 0]),
        )

    def copy_with(
        self,
        *,
        enabled: bool | object = _UNSET,
        shrink_factors: list[int] | object = _UNSET,
        smoothing_sigmas: list[float] | object = _UNSET,
    ) -> Self:
        return replace(
            self,
            enabled=(
                self.enabled
                if enabled is _UNSET
                else enabled
            ),
            shrink_factors=(
                self.shrink_factors
                if shrink_factors is _UNSET
                else shrink_factors
            ),
            smoothing_sigmas=(
                self.smoothing_sigmas
                if smoothing_sigmas is _UNSET
                else smoothing_sigmas
            ),
        )
