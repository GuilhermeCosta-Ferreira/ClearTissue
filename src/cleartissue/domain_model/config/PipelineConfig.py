# ================================================================
# 0. Section: IMPORTS
# ================================================================
from __future__ import annotations

from dataclasses import dataclass
from copy import deepcopy
from typing import Any, ClassVar, Mapping, cast

from .StepConfig import StepConfig



# ================================================================
# 1. Section: Classes
# ================================================================
@dataclass(frozen=True)
class PipelineConfig:
    config: Mapping[str, Any]

    RESERVED_KEYS: ClassVar[set[str]] = {
        "mouse",
        "tissue_type",
        "pipeline_id",
        "pipeline_name",
        "drive_root",
        "zarr_path",
    }

    def __post_init__(self) -> None:
        self.validate()



    # ================================================================
    # 2. Section: Validation
    # ================================================================
    def validate(self) -> None:
        self._require_key("pipeline_id")
        self._require_key("pipeline_name")

        if not isinstance(self.pipeline_id, int):
            raise TypeError(
                f"pipeline_id must be an int, got {type(self.pipeline_id).__name__}"
            )

        if not isinstance(self.pipeline_name, str):
            raise TypeError(
                f"pipeline_name must be a str, got {type(self.pipeline_name).__name__}"
            )

        if self.pipeline_name == "":
            raise ValueError("pipeline_name cannot be empty")

        for step_name in self.step_names:
            step_params = self.step_params(step_name)

            if not isinstance(step_params, dict):
                raise TypeError(
                    f"Config for step '{step_name}' must be a dict, "
                    f"got {type(step_params).__name__}"
                )

    def _require_key(self, key: str) -> None:
        if key not in self.config:
            raise ValueError(f"{key} is required")

        if self.config[key] is None:
            raise ValueError(f"{key} cannot be None")



    # ================================================================
    # 3. Section: Identifiers
    # ================================================================
    @property
    def mouse(self) -> str:
        return cast(str, self.config.get("mouse", ""))

    @property
    def tissue_type(self) -> str:
        return cast(str, self.config.get("tissue_type", ""))

    @property
    def pipeline_id(self) -> int:
        return cast(int, self.config["pipeline_id"])

    @property
    def pipeline_name(self) -> str:
        return cast(str, self.config["pipeline_name"])



    # ================================================================
    # 4. Section: Raw data loading config
    # ================================================================
    @property
    def drive_root(self) -> str:
        return cast(str, self.config.get("drive_root", ""))

    @property
    def zarr_path(self) -> str:
        return cast(str, self.config.get("zarr_path", ""))



    # ================================================================
    # 5. Section: Transformation config
    # ================================================================
    @property
    def step_names(self) -> list[str]:
        return [
            key
            for key in self.config.keys()
            if key not in self.RESERVED_KEYS
        ]

    def has_step(self, step_name: str) -> bool:
        return step_name in self.config

    def step_params(self, step_name: str) -> dict[str, Any]:
        if step_name not in self.config:
            raise KeyError(f"No config found for step '{step_name}'")

        params = self.config[step_name]

        if params is None:
            return {}

        if not isinstance(params, dict):
            raise TypeError(
                f"Config for step '{step_name}' must be a dict, "
                f"got {type(params).__name__}"
            )

        return deepcopy(params)

    def step_config(self, step_name: str) -> StepConfig:
        params = self.step_params(step_name)

        return StepConfig(
            name=step_name,
            enabled=False, # Everything is disabled by default
            parameters=params,
        )

    def step_configs(self, ordered_step_names: list[str]) -> list[StepConfig]:
        return [
            self.step_config(step_name)
            for step_name in ordered_step_names
            if self.has_step(step_name)
        ]



    # ================================================================
    # 6. Section: Export
    # ================================================================
    def to_dict(self) -> dict[str, Any]:
        return deepcopy(dict(self.config))
