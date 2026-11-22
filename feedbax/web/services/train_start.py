"""Where a Train step starts: from its Model with its own definition, or from a checkpoint of another Train step's runs
as that step's continuation. Which runs and which checkpoint are the selection's, and the planner judges them."""

from dataclasses import dataclass
from typing import Any

from feedbax.contracts.project import Project, RemoveProjectValue, SetProjectValue, Step
from feedbax.contracts.project_inputs import StepReference
from feedbax.contracts.project_resolution import StepTemplate, compose_step
from feedbax.contracts.project_sweep import Sweep
from feedbax.contracts.run_matrix import MatrixCompositionDelta
from feedbax.contracts.studio_authoring import StudioTrainStartEdit, StudioTrainStartRequest
from feedbax.contracts.studio_project import AuthoredFileAddress, StudioProjectFilePatch
from feedbax.studio.step_differences import difference_patches
from feedbax.web.services.project_folder import ProjectFolderInvalid, ProjectFolderService

_POLICY = "checkpoint_initialization"
_LAYER = "trial-authoring"


@dataclass(frozen=True)
class _Definition:
    payload: dict[str, Any]
    sweep: dict[str, Any] | None
    inputs: dict[str, Any]
    requires: tuple[str, ...]


def _definition(project: Project, step_id: str) -> _Definition:
    """The step's composed definition."""
    template = compose_step(project, step_id)
    if not isinstance(template, StepTemplate):
        raise ProjectFolderInvalid(template.detail)
    return _Definition(
        template.authored_payload(),
        template.sweep.model_dump(mode="json", by_alias=True) if template.sweep is not None else None,
        {name: binding.model_dump(mode="json", by_alias=True) for name, binding in template.inputs.items()},
        template.requires,
    )


def _patch_set(parent: dict[str, Any], own: dict[str, Any]) -> dict[str, Any]:
    """``own`` as an entry over ``parent``'s: the patches ``own`` lacks removed, its other fields written whole."""
    patches = own.get("patches", {})
    return own | {"patches": patches | {path: None for path in parent.get("patches", {}) if path not in patches}}


def _named(parent: dict[str, Any], own: dict[str, Any], entry) -> dict[str, Any]:
    """The entries of ``own`` that differ from ``parent``'s, and the removal of those it lacks."""
    return {
        **{name: entry(parent[name], value) if name in parent else value
           for name, value in own.items() if parent.get(name) != value},
        **{name: None for name in parent if name not in own},
    }


def _axis(parent: dict[str, Any], own: dict[str, Any]) -> dict[str, Any]:
    if "values" in parent and "values" in own and isinstance(parent["values"], dict) and isinstance(own["values"], dict):
        return own | {"values": _named(parent["values"], own["values"], _patch_set)}
    return own


def _sweep_difference(parent: dict[str, Any] | None, own: dict[str, Any] | None) -> dict[str, Any] | None:
    """The sweep entries that, over ``parent``'s sweep, give ``own``."""
    parent = parent or Sweep().model_dump(mode="json", by_alias=True)
    own = own or Sweep().model_dump(mode="json", by_alias=True)
    difference = {
        key: value for key, value in {
            "axes": _named(parent["axes"], own["axes"], _axis),
            "rows": _named(parent["rows"], own["rows"], _patch_set),
            "exceptions": _named(parent["exceptions"], own["exceptions"], _patch_set),
        }.items() if value
    }
    if parent.get("combination") != own.get("combination"):
        difference["combination"] = own.get("combination")
    return difference or None


def _started(own: _Definition, payload: dict[str, Any], source: StepReference | None, project: Project):
    """The step's authored fields that give it ``payload``, its sweep, inputs and requirements from ``source``."""
    inputs = {name: binding for name, binding in own.inputs.items() if name != "checkpoint"}
    if source is None:
        return {
            "base": {"inline": payload}, "sweep": own.sweep, "inputs": inputs or None,
            "requires": list(own.requires) or None, "deltas": None,
        }
    parent = _definition(project, source.step)
    patches = difference_patches(parent.payload, payload)
    layer = MatrixCompositionDelta(
        layer_id=_LAYER, patches=patches, acknowledges_ancestor_paths=[patch.path for patch in patches],
    )
    return {
        "base": {"step": source.step},
        "deltas": [layer.model_dump(mode="json", exclude_unset=True)] if patches else None,
        "sweep": _sweep_difference(parent.sweep, own.sweep),
        "inputs": _named(
            {name: binding for name, binding in parent.inputs.items() if name != "checkpoint"}, inputs, lambda _, own: own,
        ) | {"checkpoint": source.model_dump(mode="json", by_alias=True, exclude_none=True)},
        "requires": [path for path in own.requires if path not in parent.requires] or None,
    }


def prepare_train_start(
    folders: ProjectFolderService, project_id: str, step_id: str, body: StudioTrainStartRequest,
) -> StudioTrainStartEdit:
    """The edit that starts the Train step from ``body.source``'s checkpoint, or from its Model.

    Started from a checkpoint, the step continues that step and stores only what differs from it; started from its
    Model, it holds its whole definition, document references kept. Either way its runs keep their definition.
    """
    snapshot = folders.read(project_id, revision=body.revision)
    project = snapshot.project
    step = project.steps.get(step_id)
    source = body.source
    if step is None or step.kind != "train":
        raise ProjectFolderInvalid("Choose a Train step.")
    if source is not None and (
        (parent := project.steps.get(source.step)) is None or parent.kind != "train" or source.step == step_id
    ):
        raise ProjectFolderInvalid("A Train step starts from another Train step's checkpoint.")
    own = _definition(project, step_id)
    payload = {key: value for key, value in own.payload.items() if key != _POLICY}
    if source is not None:
        payload[_POLICY] = "initialize_from"
    raw = step.model_dump(mode="json", by_alias=True, exclude_unset=True)
    fields = _started(own, payload, source, project)
    authored = {key: value for key, value in raw.items() if key not in fields} | {
        key: value for key, value in fields.items() if value is not None
    }
    started = project.model_copy(update={"steps": project.steps | {step_id: Step.model_validate(authored)}})
    result = _definition(started, step_id)
    intended = {key: value for key, value in own.inputs.items() if key != "checkpoint"} | (
        {"checkpoint": source.model_dump(mode="json", by_alias=True)} if source is not None else {}
    )
    if (
        (result.payload, result.sweep, result.requires) != (payload, own.sweep, own.requires)
        or result.inputs != intended
    ):
        raise ProjectFolderInvalid(f"Train step {step_id!r} cannot keep its definition with this start.")
    order = list(raw)
    removed = sorted((key for key in fields if key in raw and fields[key] is None), key=order.index, reverse=True)
    patches: list[SetProjectValue | RemoveProjectValue] = [
        RemoveProjectValue(op="remove", path=f"/{key}") for key in removed
    ]
    patches.extend(
        SetProjectValue(op="replace" if key in raw else "add", path=f"/{key}", value=value)
        for key, value in fields.items() if value is not None and raw.get(key) != value
    )
    return StudioTrainStartEdit(revision=snapshot.revision, files=[
        StudioProjectFilePatch(address=AuthoredFileAddress(kind="step", id=step_id), patches=patches),
    ] if patches else [])
