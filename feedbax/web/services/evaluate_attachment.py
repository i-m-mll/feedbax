"""Prepare an addressed intervention edit on a saved Evaluate step."""

from __future__ import annotations

from hashlib import sha256

from feedbax.compiler.model_document import lower_model_document
from feedbax.contracts.canonical_json import canonical_json_v2_bytes
from feedbax.contracts.graph import GraphSpec, StudioTaskBindingSpec
from feedbax.contracts.model_address import AttachmentElement, PortElement, WireElement
from feedbax.contracts.model_document import ModelDocument
from feedbax.contracts.model_edit import (
    EvaluateAttachmentAxis, EvaluateAttachmentEdit, EvaluateAttachmentRequest, EvaluateAttachmentRows,
    ModelEditProblem,
)
from feedbax.contracts.project import Project
from feedbax.contracts.project_plan import RunKey, compile_project
from feedbax.contracts.project_inputs import ProjectName
from feedbax.contracts.request_identity import decode_request
from feedbax.contracts.rollout import RolloutParameters
from feedbax.contracts.studio_project import AuthoredFileAddress, SetProjectValue, StudioProjectFilePatch
from feedbax.studio.schema import validate_graph_connection_schema
from feedbax.training.evaluation_variant import resolve_evaluation_variant


class EvaluateAttachmentRefused(ValueError):
    def __init__(self, problem: ModelEditProblem):
        super().__init__(problem.detail)
        self.problem = problem


def _add_intervention(model, body, lowered):
    owner = model
    for component in body.owner.composites:
        child = owner["interior"]["components"].get(component)
        if child is None or child["kind"] != "composite":
            raise ValueError("Intervention address no longer names a composite")
        owner = child
    graph = owner["interior"]
    element = body.owner.element
    if isinstance(element, WireElement):
        wire = graph["wires"].get(element.id)
        if wire is None:
            raise ValueError("Intervention address no longer names a wire")
        chain = wire["attachments"]
    elif isinstance(element, PortElement):
        end = element.end
        if end.owner == "inputs":
            if not any(port["id"] == end.port for port in owner["ports"]["inputs"]):
                raise ValueError("Intervention address no longer names an input source")
            chain = graph["attachments"]["sources"].setdefault(end.port, [])
        elif end.owner == "component" and end.direction == "output":
            scoped = lowered.graph
            for name in body.owner.composites:
                scoped = scoped.subgraphs[name]
            if end.component not in scoped.nodes or end.port not in scoped.nodes[end.component].output_ports:
                raise ValueError("Intervention address no longer names a component output")
            chain = graph["components"][end.component]["attachments"]["sources"].setdefault(end.port, [])
        else:
            raise ValueError("Interventions attach to a wire or a source port")
    else:
        raise ValueError("Interventions attach to a wire or a source port")
    if body.index > len(chain) or any(item["id"] == body.attachment.id for item in chain):
        raise ValueError("The attachment index or ID conflicts with the current chain")
    chain.insert(body.index, body.attachment.model_dump(mode="json"))


def _graph_edits(before: GraphSpec, after: GraphSpec, path=("graph",)):
    for field in ("nodes", "wires", "output_bindings"):
        old = getattr(before, field)
        new = getattr(after, field)
        if field == "wires":
            old = {wire.authored_id(): wire for wire in old}
            new = {wire.authored_id(): wire for wire in new}
        for name in sorted(old.keys() | new.keys()):
            previous, current = old.get(name), new.get(name)
            if previous == current:
                continue
            parts = (*path, field, name)
            if any("." in part for part in parts):
                raise ValueError("This ID cannot be written as a variation patch")
            edit = {"path": ".".join(parts), "op": "add" if previous is None else "replace"}
            if current is None:
                edit["op"] = "remove"
            else:
                edit["value"] = list(current) if isinstance(current, tuple) else current.model_dump(mode="json")
            yield edit
    for name, child in (before.subgraphs or {}).items():
        if isinstance(child, GraphSpec):
            yield from _graph_edits(child, after.subgraphs[name], (*path, "subgraphs", name))


def _prior_graph_value(graph: GraphSpec, path: str):
    tokens = path.split(".")[1:]
    while tokens[0] == "subgraphs":
        graph = graph.subgraphs[tokens[1]]
        tokens = tokens[2:]
    field, name = tokens
    if field == "wires":
        return next((wire for wire in graph.wires if wire.authored_id() == name), None)
    return getattr(graph, field).get(name)


def prepare_evaluate_attachment(
    folders, project_id: ProjectName, body: EvaluateAttachmentRequest, components,
) -> EvaluateAttachmentEdit:
    key = RunKey(body.run.variant, body.run.step, body.run.row)
    snapshot = folders.read(project_id, revision=body.revision)
    try:
        return _prepare(folders, project_id, body, components, key, snapshot.project)
    except (KeyError, ValueError) as error:
        raise EvaluateAttachmentRefused(ModelEditProblem(
            address=getattr(error, "address", body.owner), detail=str(error),
        )) from error


def _prepare(folders, project_id, body, components, key, project):
    plan = folders.read_plan(project_id, revision=body.revision).plan
    selected = next((run for run in plan.runs if run.key == key and run.kind == "evaluate"), None)
    if selected is None or "subject" not in selected.inputs:
        raise ValueError("Select an Evaluate run with a planned Train subject")
    producer = selected.inputs["subject"].producer
    source_plan = folders.read_plan(project_id, revision=body.revision, source_runs=(producer,)).plan
    definition = source_plan.selected_definitions[producer]
    if definition.document_ids.get("payload.graph") != body.owner.document:
        raise ValueError("The Evaluate subject does not use the addressed Model document")
    model = ModelDocument.model_validate(definition.payload["graph"])
    binding = StudioTaskBindingSpec.model_validate(definition.payload.get("task_binding_spec") or {})
    before = lower_model_document(body.owner.document, model, components, binding)
    draft = model.model_dump(mode="json")
    _add_intervention(draft, body, before)
    after = lower_model_document(body.owner.document, ModelDocument.model_validate(draft), components, binding)
    edits = tuple(_graph_edits(before.graph, after.graph))
    if not edits:
        raise ValueError("The intervention has no connected signal reader")
    operation = sha256(canonical_json_v2_bytes([body.owner.to_text(), body.attachment.id])).hexdigest()[:20]
    step = project.steps[key.step].model_dump(mode="json", by_alias=True, exclude_unset=True)
    scope = body.scope
    run_spec = decode_request(plan.requests[selected.request_id], plan.documents.__getitem__).spec
    existing = RolloutParameters.model_validate(run_spec.params).variation
    empty_ordered = isinstance(existing, tuple) and not existing
    if not isinstance(existing, dict):
        if existing:
            raise ValueError("Convert the ordered Evaluate variation to named patches before adding an intervention")
        existing = {}
    path = "params.variation"
    additions = {f"attachment-{operation}-{index}": edit for index, edit in enumerate(edits)}
    if existing.keys() & additions.keys():
        raise ValueError("The selected scope already contains this attachment")
    scoped_patches = (
        {path: {"op": "replace", "value": additions}} if empty_ordered else
        {f"{path}.{name}": {"op": "add", "value": edit} for name, edit in additions.items()}
    )
    affected = {run.key for run in plan.runs if run.key.step == key.step}
    if isinstance(scope, EvaluateAttachmentAxis):
        affected = {
            run.key for run in plan.runs if run.key in affected
            and scope.axis in run.coordinates and run.coordinates[scope.axis].value_id == scope.value
        }
        axis = step["sweep"]["axes"][scope.axis]["values"][scope.value]
        patches = axis.setdefault("patches", {})
        if patches.keys() & scoped_patches.keys():
            raise ValueError("The selected group already writes the intervention path")
        patches.update(scoped_patches)
    elif isinstance(scope, EvaluateAttachmentRows):
        if any(run.variant != key.variant for run in affected):
            raise ValueError("Select a single project variant before authoring a row exception")
        affected = {run for run in affected if run.row in scope.rows}
        if step.get("sweep") is None:
            step["sweep"] = {}
        sweep = step["sweep"]
        exceptions = sweep.setdefault("exceptions", {})
        name = f"attachment-{operation}"
        if name in exceptions:
            raise ValueError("The selected step already contains a row exception for this attachment")
        exceptions[name] = {
            "where": {"rows": list(scope.rows)},
            "patches": scoped_patches,
        }
    else:
        if "inline" not in step["base"] or step.get("deltas"):
            raise ValueError("Step-wide interventions require a direct Evaluate definition")
        params = step["base"]["inline"]["params"]
        if params.get("variation") == []:
            params["variation"] = {}
        params.setdefault("variation", {}).update(additions)
    if key not in affected:
        raise ValueError("The selected run is outside the requested edit scope")

    data = project.model_dump(mode="json", by_alias=True, exclude_unset=True)
    data["steps"][key.step] = step
    candidate = compile_project(Project.model_validate(data), components=components)
    previous = {run.key: run for run in plan.runs}
    current = {run.key: run for run in candidate.runs}
    if current.keys() != previous.keys() or {
        item for item, run in current.items() if run.request_id != previous[item].request_id
    } != affected:
        if candidate.diagnostics:
            raise ValueError(candidate.diagnostics[0].message)
        raise ValueError("Intervention patches did not resolve exactly at the selected scope")
    for item in affected:
        run = current[item]
        subject = run.inputs["subject"].member.reference
        train = decode_request(candidate.requests[subject.request_id], candidate.documents.__getitem__).spec
        evaluate = decode_request(candidate.requests[run.request_id], candidate.documents.__getitem__).spec
        if any(
            _prior_graph_value(train.graph, edit["path"]) != _prior_graph_value(before.graph, edit["path"])
            for edit in edits
        ):
            raise ValueError(f"Evaluate row {item.row!r} has a different signal at this address")
        effective = resolve_evaluation_variant(train.graph, train.task, RolloutParameters.model_validate(evaluate.params).variation)
        issues = validate_graph_connection_schema(effective.graph, "/graph", train.task_binding_spec, component_registry=components)
        if issue := next((issue for issue in issues if issue.severity == "error"), None):
            raise ValueError(issue.message)
    field = "base" if scope.kind == "step" else "sweep"
    original = project.steps[key.step]
    return EvaluateAttachmentEdit(
        revision=body.revision,
        changed=body.owner.model_copy(update={"element": AttachmentElement(
            kind="attachment", parent=body.owner.element, id=body.attachment.id,
        )}),
        file=StudioProjectFilePatch(
            address=AuthoredFileAddress(kind="step", id=key.step),
            patches=[SetProjectValue(
                op="replace" if field in original.model_fields_set else "add", path=f"/{field}", value=step[field],
            )],
        ),
    )
