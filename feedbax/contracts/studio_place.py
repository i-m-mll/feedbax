"""The place in a step's definition that a reason or problem names."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from typing import Annotated, Literal

from pydantic import Field

from feedbax.contracts.base import StrictModel
from feedbax.contracts.model_address import ModelAddress
from feedbax.contracts.project_sources import pointer


class StudioModelPlace(StrictModel):
    part: Literal["model"]
    document: str | None
    member: ModelAddress | None = Field(default=None, exclude_if=lambda value: value is None)


class StudioTaskPlace(StrictModel):
    part: Literal["task"]
    document: str | None
    parameter: str | None = Field(default=None, exclude_if=lambda value: value is None)
    initializes: str | None = Field(default=None, exclude_if=lambda value: value is None)
    """The id of the variable whose Initializes, the Model values its links start, the place names."""


class StudioObjectivePlace(StrictModel):
    part: Literal["objective"]
    document: str | None


class StudioTrainingPlace(StrictModel):
    part: Literal["training"]
    document: str | None
    setting: str | None = Field(default=None, exclude_if=lambda value: value is None)


class StudioMethodPlace(StrictModel):
    part: Literal["method"]
    document: str | None
    setting: str | None = Field(default=None, exclude_if=lambda value: value is None)
    """The dotted path below ``settings`` of the setting named, when one is."""


class StudioAnalysisPlace(StrictModel):
    part: Literal["analysis"]
    element: str


class StudioInputPlace(StrictModel):
    part: Literal["input"]
    role: str


class StudioStepPlace(StrictModel):
    part: Literal["step"]
    field: str | None = Field(default=None, exclude_if=lambda value: value is None)


StudioPlace = Annotated[
    StudioModelPlace | StudioTaskPlace | StudioObjectivePlace | StudioTrainingPlace | StudioMethodPlace
    | StudioAnalysisPlace | StudioInputPlace | StudioStepPlace,
    Field(discriminator="part"),
]

_INPUT_DOCUMENTS = {"subject": "model", "model": "model", "task": "task", "python_task": "task", "objective": "objective"}


def path_parts(path: str | None) -> tuple[str, ...]:
    """Split a diagnostic path, written as a JSON pointer or a dotted path, into its keys."""
    if not path:
        return ()
    if path.startswith("/"):
        return tuple(part.replace("~1", "/").replace("~0", "~") for part in path[1:].split("/"))
    return tuple(path.split("."))


def _document_place(
    kind: str, document: str | None, rest: Sequence[str], member: ModelAddress | None, variable: str | None,
) -> StudioPlace:
    if kind == "model":
        return StudioModelPlace(part="model", document=document, member=member)
    if kind == "task":
        return StudioTaskPlace(
            part="task", document=document,
            parameter=rest[1] if len(rest) > 1 and rest[0] == "params" else None,
            initializes=variable if rest[:1] == ("links",) else None,
        )
    return StudioObjectivePlace(part="objective", document=document)


def definition_place(
    parts: Sequence[str],
    document: Callable[[str], str | None],
    *,
    element: str | None = None,
    member: ModelAddress | None = None,
    variable: str | None = None,
    variation: Callable[[str], str | None] = lambda _name: None,
) -> StudioPlace:
    """Name the part of a step's definition at ``parts``, a path from the step's root or its payload.

    ``document`` answers which project document supplies the definition at a slot such as
    ``payload.trial_documents.objective`` or ``payload.settings``, or ``None`` when the step holds it inline.
    ``variable`` names the Task variable a link at ``parts`` belongs to, so the link is placed in its Initializes.
    ``variation`` answers the dotted path a named variation entry patches, so the entry is placed
    where its field lives.
    """
    if element is not None:
        return StudioAnalysisPlace(part="analysis", element=element)
    parts = tuple(parts)
    if parts[:1] == ("inputs",) and len(parts) > 1:
        role, rest = parts[1], parts[2:]
        kind = _INPUT_DOCUMENTS.get(role)
        if kind is not None and (rest or member is not None):
            return _document_place(kind, document(f"inputs.{role}"), rest, member, variable)
        return StudioInputPlace(part="input", role=role)
    if parts[:1] == ("payload",):
        parts = parts[1:]
    field, rest = (parts[0], parts[1:]) if parts else (None, ())
    match field:
        case "trial_documents" if rest:
            role = rest[0]
            kind = "model" if role == "model" else "task" if role in ("task", "python_task") else "objective"
            return _document_place(kind, document(f"payload.trial_documents.{role}"), rest[1:], member, variable)
        case "graph":
            return StudioModelPlace(part="model", document=document("payload.graph"), member=member)
        case "task" | "task_binding_spec":
            return _document_place("task", document(f"payload.{field}"), rest, member, variable)
        case "objective":
            return StudioObjectivePlace(part="objective", document=document("payload.objective"))
        case "training_config":
            return StudioTrainingPlace(
                part="training", document=document("payload.training_config"), setting=rest[0] if rest else None,
            )
        case "method" | "settings":
            return StudioMethodPlace(
                part="method", document=document(f"payload.{field}"), setting=".".join(rest) or None,
            )
        case "params" if rest[:1] == ("variation",):
            target = path_parts(variation(rest[1])) if len(rest) > 1 else ()
            if target[:1] == ("task",):
                return _document_place("task", document("payload.task"), target[1:], member, variable)
            return StudioModelPlace(part="model", document=None, member=member)
    if member is not None:
        model = document("payload.trial_documents.model") or document("payload.graph")
        return StudioModelPlace(part="model", document=model, member=member)
    return StudioStepPlace(part="step", field=field)


def document_path(parts: Sequence[str], place: StudioPlace) -> str | None:
    """The JSON pointer, inside the document ``place`` names, of a trial document's member at ``parts``."""
    if not isinstance(place, StudioModelPlace | StudioTaskPlace | StudioObjectivePlace) or place.document is None:
        return None
    parts = tuple(parts)
    if parts[:1] == ("payload",):
        parts = parts[1:]
    if parts[:1] not in (("inputs",), ("trial_documents",)) or len(parts) < 2:
        return None
    return pointer(parts[2:])


def variation_targets(params: object) -> Callable[[str], str | None]:
    """The dotted path each named entry of a request's ``params.variation`` patches."""
    variation = params.get("variation") if isinstance(params, Mapping) else None

    def target(name: str) -> str | None:
        if isinstance(variation, Mapping):
            edit = variation.get(name)
        elif isinstance(variation, list) and name.isdigit() and int(name) < len(variation):
            edit = variation[int(name)]
        else:
            return None
        path = edit.get("path") if isinstance(edit, Mapping) else None
        return path if isinstance(path, str) else None

    return target
