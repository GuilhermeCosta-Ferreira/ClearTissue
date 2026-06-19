# ================================================================
# 0. Section: IMPORTS
# ================================================================
from typing import Self, Any
from dataclasses import dataclass, field, replace

from .config import (
    MetricConfig,
    OptimizerConfig,
    InterpolationConfig,
    MultipleResolutionConfig,
)

_UNSET = object()



# ================================================================
# 1. Section: Functions
# ================================================================
@dataclass
class RegistrationConfig:
    seed: int = 7
    transform_center: int = 1  # 1 is Moments
    metric: MetricConfig = field(default_factory=MetricConfig)
    optimizer: OptimizerConfig = field(default_factory=OptimizerConfig)
    interpolator: InterpolationConfig = field(default_factory=InterpolationConfig)
    multiple_res: MultipleResolutionConfig = field(
        default_factory=MultipleResolutionConfig
    )

    @classmethod
    def from_dict(cls, data: dict) -> "RegistrationConfig":
        return cls(
            seed=data.get("seed", 7),
            transform_center=data.get("transform_center", 1),
            metric=MetricConfig.from_dict(data.get("MetricConfig", {})),
            optimizer=OptimizerConfig.from_dict(data.get("OptimizerConfig", {})),
            interpolator=InterpolationConfig.from_dict(data.get("InterpolationConfig", {})),
            multiple_res=MultipleResolutionConfig.from_dict(data.get("MultipleResolutionConfig", {})),
        )

    def copy_with(
        self,
        *,
        seed: int | object = _UNSET,
        transform_center: int | object = _UNSET,
        metric: MetricConfig | object = _UNSET,
        optimizer: OptimizerConfig | object = _UNSET,
        interpolator: InterpolationConfig | object = _UNSET,
        multiple_res: MultipleResolutionConfig | object = _UNSET,
    ) -> Self:
        return replace(
            self,
            seed=self.seed if seed is _UNSET else seed,
            transform_center=(
                self.transform_center
                if transform_center is _UNSET
                else transform_center
            ),
            metric=self.metric if metric is _UNSET else metric,
            optimizer=self.optimizer if optimizer is _UNSET else optimizer,
            interpolator=(
                self.interpolator
                if interpolator is _UNSET
                else interpolator
            ),
            multiple_res=(
                self.multiple_res
                if multiple_res is _UNSET
                else multiple_res
            ),
        )

    def with_overrides(self, data: dict[str, Any]) -> Self:
        seed = data.get("seed", _UNSET)
        transform_center = data.get("transform_center", _UNSET)

        metric = (
            self.metric.copy_with(**data["MetricConfig"])
            if "MetricConfig" in data
            else _UNSET
        )

        optimizer = (
            self.optimizer.copy_with(**data["OptimizerConfig"])
            if "OptimizerConfig" in data
            else _UNSET
        )

        interpolator = (
            self.interpolator.copy_with(**data["InterpolationConfig"])
            if "InterpolationConfig" in data
            else _UNSET
        )

        multiple_res = (
            self.multiple_res.copy_with(**data["MultipleResolutionConfig"])
            if "MultipleResolutionConfig" in data
            else _UNSET
        )

        return self.copy_with(
            seed=seed,
            transform_center=transform_center,
            metric=metric,
            optimizer=optimizer,
            interpolator=interpolator,
            multiple_res=multiple_res,
        )
