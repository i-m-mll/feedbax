"""Read the recorded upstream Train for a selected Evaluate definition."""

from feedbax.contracts.model_edit import StudioEvaluateValueSource
from feedbax.contracts.project_plan import RunKey, SelectedRunDefinition
from feedbax.contracts.project_sources import ValueSource


def evaluate_subject(definition: SelectedRunDefinition | None) -> tuple[RunKey, ValueSource] | None:
    source = definition.sources.get("/inputs/subject") if definition is not None else None
    if source is None or source.kind != "upstream" or not (source.variant and source.step and source.row):
        return None
    return RunKey(source.variant, source.step, source.row), source


def evaluate_value_source(source: ValueSource, *, evaluate_step: str | None = None) -> StudioEvaluateValueSource:
    if evaluate_step is not None:
        return StudioEvaluateValueSource(kind="evaluate", step=source.step or evaluate_step)
    return StudioEvaluateValueSource(kind="trained", step=source.step or source.name)
