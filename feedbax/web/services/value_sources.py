"""Display names for exact authored value origins."""

from collections.abc import Mapping
from functools import cache

from feedbax.contracts.jobs import StudioPlanReviewSource
from feedbax.contracts.member_family import family_member_count
from feedbax.contracts.project import Project, json_pointer_parts, member_name
from feedbax.contracts.project_sources import ValueSource
from feedbax.contracts.project_sweep import SweepCoordinate, axis_coordinates, sweep_value_labels
from feedbax.contracts.set_document import SET_DOCUMENT_SCHEMA_ID, read_set
from feedbax.planner.project_plan import PlanSourceIndex, ProjectPlan, RunKey


class PlanSourceNames:
    def __init__(self, plan: ProjectPlan | PlanSourceIndex) -> None:
        self.plan = plan
        self._coordinates = cache(self._step_coordinates)
        self._set = cache(lambda variant, key: read_set(self._project(variant).documents.get, key))

    def _project(self, variant: str) -> Project:
        if isinstance(self.plan, ProjectPlan):
            return self.plan.variant_projects[variant]
        return self.plan.variant_project(variant)

    def _step_coordinates(self, variant: str, step: str) -> dict[str, Mapping[str, SweepCoordinate]]:
        if isinstance(self.plan, ProjectPlan):
            return {
                row.key.row: row.selection_coordinates for row in self.plan.rows
                if (row.key.variant, row.key.step) == (variant, step)
            }
        _, rows = self.plan.expanded_step(variant, step)
        return {key: axis_coordinates(row.coordinates, row.input_axes, row.kept_axes) for key, row in rows.items()}

    def _coordinate(self, key: RunKey, source: ValueSource) -> SweepCoordinate | None:
        assert source.axis is not None
        variant = source.variant or key.variant
        if variant == key.variant:
            here = self._coordinates(variant, key.step).get(key.row, {}).get(source.axis)
            if here is not None and (source.step == key.step or here.value_id == source.value):
                return here
        if source.step is None:
            return None
        return next((
            coordinate for row in self._coordinates(variant, source.step).values()
            if (coordinate := row.get(source.axis)) is not None and coordinate.value_id == source.value
        ), None)

    def project(self, source: ValueSource, run: RunKey) -> StudioPlanReviewSource:
        """Attach names without changing the authored origin or its coordinates."""
        variant = source.variant or run.variant
        project = self._project(variant)
        if source.kind == "document":
            document = project.documents.get(source.name)
            if (
                document is not None and document.schema_id == SET_DOCUMENT_SCHEMA_ID
                and source.axis is not None and source.value is not None
                and json_pointer_parts(source.path)[:3] == ("entries", source.value, "differences")
            ):
                entry = self._set(variant, source.name).entries[source.value]
                step = project.steps.get(source.step) if source.step is not None else None
                axis = step.sweep.axes.get(source.axis) if step is not None and step.sweep is not None else None
                return StudioPlanReviewSource.from_source(
                    source, kind="sweep_value", name=entry.label,
                    axis_name=(axis.label if axis is not None else None) or document.label,
                    member_count=family_member_count(source.value, entry) if entry.over is not None else None,
                )
            name = document.label if document is not None else source.name
        elif source.kind == "sweep_value":
            coordinate = self._coordinate(run, source) if source.axis is not None else None
            name = sweep_value_labels(coordinate)[1] if coordinate is not None else "Sweep value"
            return StudioPlanReviewSource.from_source(
                source, name=name, axis_name=coordinate.axis_label if coordinate is not None else "Sweep",
            )
        elif source.kind == "variant":
            name = self.plan.variant_labels[variant] if isinstance(self.plan, ProjectPlan) else self.plan.variant_label(variant)
        elif source.kind == "exception":
            name = "Exception"
        else:
            name = member_name(project.steps, source.step or source.name)
        return StudioPlanReviewSource.from_source(source, name=name)
