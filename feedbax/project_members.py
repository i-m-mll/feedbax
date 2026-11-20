"""Find and rewrite every authored reference to a keyed project member, and set members' labels."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from copy import deepcopy
import json
from typing import Any, Literal

from pydantic import JsonValue

from feedbax.contracts.project import (
    BasedDocument, Document, Project, RootDocument, diff, json_pointer_parts, nearest_shared_base,
)
from feedbax.contracts.project_inputs import DocumentReference
from feedbax.contracts.project_plan import ProjectPlan
from feedbax.contracts.project_resolution import ProjectDiagnostic, compose_document
from feedbax.contracts.project_sources import pointer
from feedbax.contracts.result_address import (
    ANALYSIS_SCHEMA, ELEMENT_FAMILIES, family_documents, payload_elements, project_holders,
)
from feedbax.contracts.studio_project import KeyedFamily, MemberFamily

type _Json = Any
type _Rename = Callable[[str], str]


def generated_name(base: str, taken: Callable[[str], bool]) -> str:
    """The name Studio gives a member it creates: ``base`` while it is free, then the first free ``base-2``, ``base-3``.

    The rule of the interface's ``generatedName`` (``web/src/features/projectFolder/generatedName.ts``).
    """
    name, number = base, 2
    while taken(name):
        name, number = f"{base}-{number}", number + 1
    return name


def _renamed_prefix(family: str, old: str, new: str) -> _Rename:
    """Rewrite a sweep target that addresses the member by name, `<family>.<name>...`."""
    prefix = f"{family}.{old}"

    def rename(path: str) -> str:
        if path == prefix or path.startswith(prefix + "."):
            return f"{family}.{new}" + path.removeprefix(prefix)
        return path

    return rename


def _renamed_patch_targets(patchset: _Json, rename: _Rename) -> _Json:
    if patchset is None or "patches" not in patchset:
        return patchset
    return {**patchset, "patches": {rename(path): patch for path, patch in patchset["patches"].items()}}


def _copied_patch_targets(patchset: _Json, copy: _Rename) -> _Json:
    """Add, beside each patch whose target ``copy`` rewrites, the same patch at the rewritten target."""
    if patchset is None or "patches" not in patchset:
        return patchset
    patches = patchset["patches"]
    return {**patchset, "patches": {**patches, **{copy(path): patch for path, patch in patches.items() if copy(path) != path}}}


def _mapped_sweep(sweep: _Json, patchset: Callable[[_Json], _Json], path: _Rename) -> _Json:
    """Map the patch sets of a sweep's rows, exceptions and value axes, and the paths of its path axes."""
    if sweep is None:
        return sweep
    sweep = dict(sweep)
    for section in ("rows", "exceptions"):
        if section in sweep:
            sweep[section] = {name: patchset(item) for name, item in sweep[section].items()}
    axes = {}
    for name, axis in sweep.get("axes", {}).items():
        if axis is not None and "path" in axis:
            axis = {**axis, "path": path(axis["path"])}
        elif axis is not None and isinstance(values := axis.get("values"), dict):
            axis = {**axis, "values": {key: patchset(item) for key, item in values.items()}}
        axes[name] = axis
    if "axes" in sweep:
        sweep["axes"] = axes
    return sweep


def _renamed_sweep(sweep: _Json, rename: _Rename) -> _Json:
    """Rewrite the targets of a sweep's rows, exceptions and axes, the only places targets are authored."""
    return _mapped_sweep(sweep, lambda item: _renamed_patch_targets(item, rename), rename)


def _renamed_sweeps(project: dict[str, JsonValue], rename: _Rename) -> dict[str, JsonValue]:
    """Rewrite the targets of the project's sweep (its variants) and of every step's sweep."""
    project = dict(project)
    if "sweep" in project:
        project["sweep"] = _renamed_sweep(project["sweep"], rename)
    if "steps" in project:
        project["steps"] = {
            name: {**step, "sweep": _renamed_sweep(step["sweep"], rename)} if "sweep" in step else step
            for name, step in project["steps"].items()
        }
    return project


def _renamed_document_references(value: _Json, old: str, new: str) -> _Json:
    """Rewrite every `DocumentReference` to the document, at any depth of a binding."""
    if isinstance(value, dict):
        if value.get("document") == old and value.keys() <= DocumentReference.model_fields.keys():
            return {**value, "document": new}
        return {key: _renamed_document_references(item, old, new) for key, item in value.items()}
    if isinstance(value, list):
        return [_renamed_document_references(item, old, new) for item in value]
    return value


def _renamed_step_references(value: _Json, old: str, new: str) -> _Json:
    """Rewrite `StepBase.step` and `StepReference.step`, at any depth of a binding."""
    if isinstance(value, dict):
        return {
            key: new if key == "step" and item == old else _renamed_step_references(item, old, new)
            for key, item in value.items()
        }
    if isinstance(value, list):
        return [_renamed_step_references(item, old, new) for item in value]
    return value


def _rekey(mapping: dict[str, _Json], old: str, new: str) -> dict[str, _Json]:
    return {new if key == old else key: item for key, item in mapping.items()}


def _rename_document(
    project: dict[str, JsonValue], presentation: dict[str, JsonValue], old: str, new: str,
) -> tuple[dict[str, JsonValue], dict[str, JsonValue]]:
    renamed = _renamed_document_references({key: value for key, value in project.items() if key != "documents"}, old, new)
    renamed = _renamed_sweeps(renamed, _renamed_prefix("documents", old, new))
    documents = {
        name: {**document, "base": {"document": new}} if document.get("base") == {"document": old} else document
        for name, document in project.get("documents", {}).items()
    }
    renamed = {"documents": _rekey(documents, old, new), **renamed}
    shown = dict(presentation)
    if "documents" in shown:
        shown["documents"] = _rekey(shown["documents"], old, new)
    if "steps" in shown:
        shown["steps"] = {
            name: {**step, "model_positions": _rekey(step["model_positions"], old, new)}
            if "model_positions" in step else step
            for name, step in shown["steps"].items()
        }
    return renamed, shown


def _rename_step(
    project: dict[str, JsonValue], presentation: dict[str, JsonValue], old: str, new: str,
) -> tuple[dict[str, JsonValue], dict[str, JsonValue]]:
    renamed = _renamed_sweeps(project, _renamed_prefix("steps", old, new))
    steps = {}
    for name, step in renamed.get("steps", {}).items():
        step = dict(step)
        for field in ("base", "inputs"):
            if field in step:
                step[field] = _renamed_step_references(step[field], old, new)
        steps[new if name == old else name] = step
    if "steps" in renamed:
        renamed["steps"] = steps
    shown = dict(presentation)
    if "steps" in shown:
        shown["steps"] = _rekey(shown["steps"], old, new)
    return renamed, shown


def change_key(
    project: dict[str, JsonValue], presentation: dict[str, JsonValue],
    family: MemberFamily, old: str, new: str,
) -> tuple[dict[str, JsonValue], dict[str, JsonValue]]:
    """Return the project and presentation with the document or step keyed ``new`` and its references rewritten.

    A document is referred to by the project's `DocumentReference`s, the `base` of the
    documents based on it, and `documents.<name>...` sweep targets. Documents' content is left byte for byte, Model
    addresses are relative to their document and carry no name, and Model difference batches are named by their
    author. A step is referred to by `StepBase.step`, `StepReference.step` in other
    steps' inputs, and `steps.<name>...` targets in the project's and steps' sweeps. A step's group membership is its own `group`
    field, so it moves with the step. The presentation is keyed by member name.
    """
    rename = _rename_document if family == "documents" else _rename_step
    return rename(project, presentation, old, new)


def _wires_renamed(value: _Json, old: str, new: str) -> _Json:
    """Rewrite every operation dataset wire `{element, product}` from ``old`` at any depth."""
    if isinstance(value, dict):
        if value.keys() == {"element", "product"} and value["element"] == old:
            return {**value, "element": new}
        return {key: _wires_renamed(item, old, new) for key, item in value.items()}
    if isinstance(value, list):
        return [_wires_renamed(item, old, new) for item in value]
    return value


def _analysis_value_renamed(parts: tuple[str, ...], value: _Json, old: str, new: str) -> _Json:
    """Rewrite the value an Analysis payload holds at ``parts`` for the element key ``old`` becoming ``new``."""
    if not parts and isinstance(value, dict):
        return {
            key: _analysis_value_renamed((key,), item, old, new) if key in ELEMENT_FAMILIES else item
            for key, item in value.items()
        }
    if len(parts) == 1 and isinstance(value, dict):
        return {new if key == old else key: _wires_renamed(item, old, new) for key, item in value.items()}
    if len(parts) >= 4 and parts[2] == "inputs" and parts[-1] == "element" and value == old:
        return new
    return _wires_renamed(value, old, new)


def _analysis_pointer_renamed(path: str, old: str, new: str) -> str:
    parts = json_pointer_parts(path)
    if len(parts) >= 2 and parts[0] in ELEMENT_FAMILIES and parts[1] == old:
        return pointer((parts[0], new, *parts[2:]))
    return path


def _renamed_prefixes(prefixes: list[str], old: str, new: str) -> _Rename:
    """Rewrite a dotted target `<prefix>.<family>.<old>...` for each prefix (empty for a step's own payload)."""
    renames = [
        _renamed_prefix(f"{prefix}{family}", old, new)
        for prefix in prefixes for family in ELEMENT_FAMILIES
    ]

    def rename(path: str) -> str:
        for each in renames:
            path = each(path)
        return path

    return rename


def _reads(steps: dict[str, _Json], name: str, documents: frozenset[str]) -> bool:
    """Whether step ``name``'s base chain reaches one of ``documents``."""
    seen = set()
    while name in steps and name not in seen:
        seen.add(name)
        base = steps[name].get("base", {})
        if base.get("document") in documents:
            return True
        name = base.get("step")
    return False


def _change_element_key(
    project: dict[str, JsonValue], presentation: dict[str, JsonValue], document: str, old: str, new: str,
) -> tuple[dict[str, JsonValue], dict[str, JsonValue]]:
    typed = Project.model_validate(project)
    family = frozenset(
        name for name in family_documents(typed.documents, document)
        if typed.documents[name].schema_id == ANALYSIS_SCHEMA
    )
    holders = project_holders(typed)
    documents = dict(project["documents"])
    for name in family:
        entry = dict(documents[name])
        if "payload" in entry:
            entry["payload"] = _analysis_value_renamed((), entry["payload"], old, new)
        else:
            entry["differences"] = {
                _analysis_pointer_renamed(path, old, new): {**change, "value": _analysis_value_renamed(
                    json_pointer_parts(path), change["value"], old, new,
                )} if "value" in change else change
                for path, change in entry.get("differences", {}).items()
            }
        documents[name] = entry
    steps = dict(project.get("steps", {}))
    readers = sorted(name for name in steps if _reads(steps, name, family))

    def references(value: _Json) -> _Json:
        if isinstance(value, dict):
            if value.get("element") == old and "product" in value and "step" in value:
                card = holders.card_of(value["step"], old)
                if card in readers:
                    return {**value, "step": card, "element": new}
            return {key: references(item) for key, item in value.items()}
        if isinstance(value, list):
            return [references(item) for item in value]
        return value

    own = _renamed_prefixes([""], old, new)
    for name, step in steps.items():
        step = {**step, "inputs": references(step["inputs"])} if "inputs" in step else dict(step)
        if name in readers:
            if "sweep" in step:
                step["sweep"] = _renamed_sweep(step["sweep"], own)
            if "requires" in step:
                step["requires"] = [own(path) for path in step["requires"]]
            if "deltas" in step:
                step["deltas"] = [{
                    **delta,
                    "patches": [{**patch, "path": own(patch["path"])} for patch in delta.get("patches", [])],
                    **({"acknowledges_ancestor_paths": [own(path) for path in delta["acknowledges_ancestor_paths"]]}
                       if "acknowledges_ancestor_paths" in delta else {}),
                } for delta in step["deltas"]]
        steps[name] = step
    renamed = {**project, "documents": documents, "steps": steps}
    if "sweep" in renamed:
        prefixes = [f"steps.{name}." for name in readers] + [f"documents.{name}." for name in sorted(family)]
        renamed["sweep"] = _renamed_sweep(renamed["sweep"], _renamed_prefixes(prefixes, old, new))
    shown = dict(presentation)
    if "documents" in shown:
        shown["documents"] = {
            name: _element_presentation(entry, old, new) if name in family else entry
            for name, entry in shown["documents"].items()
        }
    return renamed, shown


def _element_presentation(entry: dict[str, _Json], old: str, new: str) -> dict[str, _Json]:
    entry = dict(entry)
    for section in ("positions", "flipped", "collapsed"):
        if section in entry:
            entry[section] = _rekey(entry[section], old, new)
    if "waypoints" in entry:
        entry["waypoints"] = {_waypoint_renamed(key, old, new): item for key, item in entry["waypoints"].items()}
    return entry


def _waypoint_renamed(key: str, old: str, new: str) -> str:
    """A waypoint is keyed by its edge, the JSON array `[target, role]`."""
    try:
        target, role = json.loads(key)
    except (ValueError, TypeError):
        return key
    return json.dumps([new, role], separators=(",", ":")) if target == old else key


def change_member_key(
    project: dict[str, JsonValue], presentation: dict[str, JsonValue],
    family: KeyedFamily, old: str, new: str, *, document: str | None = None,
) -> tuple[dict[str, JsonValue], dict[str, JsonValue]]:
    """Return the project and presentation with the member keyed ``new`` and every reference to it rewritten.

    An operation's or figure's key is held by the family of Analysis documents ``document`` belongs to: its wires
    there, the references that resolve to a card reading that family (which are rewritten to that card), the cards'
    sweep, delta and requirement targets under `operations.<key>` or `views.<key>`, the project sweep's
    `steps.<card>...` and `documents.<document>...` targets, and the documents' presentation.
    """
    if family == "elements":
        assert document is not None
        return _change_element_key(project, presentation, document, old, new)
    return change_key(project, presentation, family, old, new)


def label_member(
    project: dict[str, JsonValue], family: Literal["steps", "elements"], member: str, label: str | None,
    *, document: str | None = None,
) -> dict[str, JsonValue]:
    """Return the project with the step's, or ``document``'s operation's or figure's, display label set or cleared.

    A based document holds the label as a difference from its base, and holds none when the base has the same label.
    """
    if family == "steps":
        step = {key: value for key, value in project["steps"][member].items() if key != "label"}
        return {**project, "steps": {**project["steps"], member: step | ({"label": label} if label else {})}}
    assert document is not None
    typed = Project.model_validate(project)
    composed = compose_document(typed.documents.get, document)
    if isinstance(composed, ProjectDiagnostic):
        raise ValueError(f"{document} cannot be composed: {composed.detail}")
    payload = deepcopy(composed.payload)
    section = next((name for name in ELEMENT_FAMILIES if member in payload.get(name, {})), None)
    if section is None:
        raise KeyError(member)
    element = {key: value for key, value in payload[section][member].items() if key != "label"}
    payload[section][member] = element | ({"label": label} if label else {})
    entry = project["documents"][document]
    if "payload" in entry:
        entry = {**entry, "payload": payload}
    else:
        base = compose_document(typed.documents.get, entry["base"]["document"])
        if isinstance(base, ProjectDiagnostic):
            raise ValueError(f"{document}'s base cannot be composed: {base.detail}")
        entry = {**entry, "differences": {
            path: change.model_dump(mode="json") for path, change in diff(base.payload, payload).items()
        }}
    return {**project, "documents": {**project["documents"], document: entry}}


def fresh_element_keys(payload: dict[str, JsonValue], taken: set[str]) -> dict[str, JsonValue]:
    """An Analysis payload whose operations and figures are given keys outside ``taken``, `<key>-2` first."""
    for old in sorted(payload_elements(payload)):
        new = generated_name(old, lambda name: name == old or name in taken)
        taken.add(new)
        payload = _analysis_value_renamed((), payload, old, new)
    return payload


def duplicate_step(
    project: dict[str, JsonValue], presentation: dict[str, JsonValue], old: str, new: str,
) -> tuple[dict[str, JsonValue], dict[str, JsonValue]]:
    """Return the project and presentation with step ``old`` copied as ``new``, reading the same documents and inputs.

    A step's own sweep addresses the step relative to itself and is copied as it is. Each
    patch the project's sweep makes to ``old`` it also makes to ``new``, so that the copy
    compiles to the same requests as its source; a project sweep axis whose path addresses
    ``old`` is one path and stays on ``old``.
    """
    copied = {**project, "steps": {**project["steps"], new: project["steps"][old]}}
    if "sweep" in project:
        retarget = _renamed_prefix("steps", old, new)
        copied["sweep"] = _mapped_sweep(project["sweep"], lambda item: _copied_patch_targets(item, retarget), lambda path: path)
    shown = dict(presentation)
    if old in shown.get("steps", {}):
        shown["steps"] = {**shown["steps"], new: shown["steps"][old]}
    return copied, shown


def references_member(project: dict[str, JsonValue], family: MemberFamily, name: str) -> bool:
    """Whether any authored reference names the member, whether or not it exists."""
    others = {**project, family: {key: value for key, value in project.get(family, {}).items() if key != name}}
    return change_key(others, {}, family, name, name + "-")[0] != others


def detach_refusals(plan: ProjectPlan, documents: Mapping[str, Document]) -> frozenset[str]:
    """The based documents whose Detach would leave an Evaluate step's Model document sharing no document with its subject's."""
    related = [pair for pair in plan.evaluated_models if nearest_shared_base(documents, *pair) is not None]
    refused = set()
    for name, document in documents.items():
        if isinstance(document, BasedDocument):
            detached = {**documents, name: RootDocument.model_construct(schema_id=document.schema_id, payload={})}
            if any(nearest_shared_base(detached, *pair) is None for pair in related):
                refused.add(name)
    return frozenset(refused)
