# ================================================================
# 0. Section: IMPORTS
# ================================================================
import json

from pathlib import Path
from dataclasses import dataclass

from .Source import Source
from ..domain_model.execution_metadata import StepManifest, StepStatus



# ================================================================
# 1. Section: Functions
# ================================================================
@dataclass
class StepManifestRepository:
    source: Source

    def __post_init__(self):
        self._manifest_name = "manifest.json"



    # ================================================================
    # 0. Section: Class Methods
    # ================================================================
    def find_by_signature(self, signature: str) -> StepManifest | None:
        for manifest_path in self._iter_manifest_paths():
            manifest = self._load_manifest_path(manifest_path)

            if manifest.signature != signature:
                continue

            resolved = self.resolve_manifest(
                pipeline_id=manifest.pipeline_id,
                step_id=manifest.step_id,
            )

            if resolved.status == StepStatus.MATERIALIZED:
                return resolved

        return None

    def save_manifest(self, manifest: StepManifest) -> None:
        step_path = self.source.step_path(
            pipeline_id=manifest.pipeline_id,
            step=manifest.step_id,
        )
        step_path.mkdir(parents=True, exist_ok=True)

        manifest_path = step_path / self._manifest_name

        with manifest_path.open("w", encoding="utf-8") as f:
            json.dump(manifest.to_dict(), f, indent=4)

    def save_redirect(
        self,
        pipeline_id: int,
        step_id: int,
        source_pipeline_id: int,
        source_step_id: int,
        signature: str,
    ) -> None:
        source_manifest = self.resolve_manifest(
            pipeline_id=source_pipeline_id,
            step_id=source_step_id,
        )

        redirect_manifest = StepManifest(
            pipeline_id=pipeline_id,
            step_id=step_id,
            step_name=source_manifest.step_name,
            signature=signature,
            status=StepStatus.REDIRECT,
            source_pipeline_id=source_manifest.pipeline_id,
            source_step_id=source_manifest.step_id,
        )

        self.save_manifest(redirect_manifest)

    def resolve_manifest(self, pipeline_id: int, step_id: int) -> StepManifest:
        manifest = self.load_manifest(pipeline_id, step_id)

        visited: set[tuple[int, int]] = set()

        while manifest.status == StepStatus.REDIRECT:
            key = (manifest.pipeline_id, manifest.step_id)

            if key in visited:
                raise RuntimeError(
                    "Circular step redirect detected: "
                    f"pipeline_id={pipeline_id}, step_id={step_id}"
                )

            visited.add(key)

            if manifest.source_pipeline_id is None or manifest.source_step_id is None:
                raise ValueError(
                    "Invalid redirect manifest: missing source pipeline or step id"
                )

            manifest = self.load_manifest(
                pipeline_id=manifest.source_pipeline_id,
                step_id=manifest.source_step_id,
            )

        return manifest

    def load_manifest(self, pipeline_id: int, step_id: int) -> StepManifest:
            manifest_path = self._manifest_path(pipeline_id, step_id)

            if not manifest_path.exists():
                raise FileNotFoundError(f"Missing step manifest: {manifest_path}")

            return self._load_manifest_path(manifest_path)


    # ──────────────────────────────────────────────────────
    # 1.1 Subsection: Helper Functions
    # ──────────────────────────────────────────────────────
    def _manifest_path(self, pipeline_id: int, step_id: int) -> Path:
        return self.source.step_path(pipeline_id, step_id) / self._manifest_name

    def _load_manifest_path(self, manifest_path: Path) -> StepManifest:
        with manifest_path.open("r", encoding="utf-8") as f:
            data = json.load(f)

        return StepManifest.from_dict(data)

    def _iter_manifest_paths(self) -> list[Path]:
        project_path = self.source.folder_path

        return list(project_path.rglob(self._manifest_name))
