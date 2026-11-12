"""Prepare a revision-bound edit to a named document or its lit step users."""

from copy import deepcopy
from typing import Any

from pydantic import JsonValue

from feedbax.component_registry import ComponentRegistry
from feedbax.contracts.graph import GraphSpec, StudioTaskBindingSpec
from feedbax.contracts.matrix_core import apply_value_patch_in_place
from feedbax.contracts.project import (
    DocumentOverrides, InlineBase, Project, apply_document_edits, document_edit_ops, json_pointer_parts,
)
from feedbax.contracts.project_plan import RunKey
from feedbax.contracts.project_sources import pointer
from feedbax.contracts.studio_authoring import (
    StudioAuthoringSource,
    StudioDocumentAuthoring,
    StudioDocumentAuthoringRequest,
    StudioModelAddComponent,
    StudioModelAuthoringChange,
    StudioModelConnectPorts,
    StudioModelExposeOutput,
)
from feedbax.contracts.studio_project import (
    PRESENTATION_ADDRESS, AuthoredFileAddress, SetProjectValue, StudioProjectFilePatch, StudioProjectSnapshot,
)
from feedbax.web.services.model_authoring import prepare_model_edit, validate_model_connections
from feedbax.web.services.project_folder import ProjectFolderInvalid, ProjectFolderService, ProjectRevisionConflict


def _step_difference(
    project: Project, step: str, document: str, local: DocumentOverrides,
) -> StudioProjectFilePatch:
    authored = project.steps[step]
    differences: dict[str, JsonValue] = authored.model_dump(mode="json")["document_overrides"]
    if not local:
        differences.pop(document, None)
    else:
        differences[document] = local if isinstance(local, dict) else [
            edit.model_dump(mode="json") for edit in local
        ]
    return StudioProjectFilePatch(
        address=AuthoredFileAddress(kind="step", id=step),
        patches=[SetProjectValue(
            op="replace" if "document_overrides" in authored.model_fields_set else "add",
            path="/document_overrides", value=differences,
        )],
    )


def _scalar_difference(
    project: Project, step: str, document: str, path: str, value: JsonValue, *, shared: bool,
) -> StudioProjectFilePatch | None:
    previous = project.steps[step].document_overrides.get(document, {})
    if isinstance(previous, dict):
        local = (
            {key: item for key, item in previous.items() if key != path}
            if shared else {**previous, path: value}
        )
    else:
        if shared:
            local = [edit for edit in previous if not (edit.op == "replace" and edit.path == path)]
        else:
            updated = SetProjectValue(op="replace", path=path, value=value)
            prior = previous[:-1] if previous and previous[-1].op == "replace" and previous[-1].path == path else previous
            local = [*prior, updated]
    if isinstance(local, list):
        raw = project.documents[document].payload
        if not isinstance(raw, dict):
            raise ProjectFolderInvalid("Document differences require an object payload")
        try:
            apply_document_edits(deepcopy(raw), local)
        except ValueError as error:
            raise ProjectFolderInvalid(f"{step} cannot edit that Model value: {error}") from error
    return _step_difference(project, step, document, local) if local != previous else None


def _step_source_path(project: Project, step: str, document: str, source: StudioAuthoringSource) -> str:
    parts = json_pointer_parts(source.path)
    if len(parts) < 3 or parts[:2] != ("document_overrides", document):
        raise ProjectFolderInvalid("This value is outside the named document.")
    edits = project.steps[step].document_overrides[document]
    if isinstance(edits, dict):
        return pointer((*json_pointer_parts(parts[2]), *parts[3:]))
    edit = edits[int(parts[2])]
    if edit.op == "remove" or parts[3:4] != ("value",):
        raise ProjectFolderInvalid("This value has no authored scalar source.")
    target = json_pointer_parts(edit.path)
    if "-" in target:
        raise ProjectFolderInvalid("Edit the Model element at its named address.")
    return pointer((*target, *parts[4:]))


def _placed_component(
    snapshot: StudioProjectSnapshot, document: str, steps: set[str], *, shared: bool,
    component: str, operation: StudioModelAddComponent,
) -> StudioProjectFilePatch:
    if snapshot.presentation is None:
        raise ProjectFolderInvalid("The project presentation could not be read.")
    assert operation.position is not None
    data = snapshot.presentation.model_dump(mode="json", exclude_unset=True)
    positions = [("documents", document, "positions", component)] if shared else [
        ("steps", step, "model_positions", document, component) for step in sorted(steps)
    ]
    edits = []
    for path in positions:
        value: JsonValue = operation.position.model_dump(mode="json")
        parent = data
        for index, part in enumerate(path):
            if part not in parent:
                for name in reversed(path[index + 1:]):
                    value = {name: value}
                op = "add"
                break
            if index == len(path) - 1:
                op = "replace"
                break
            parent = parent[part]
        target = pointer(path[:index + 1])
        apply_value_patch_in_place(data, path[:index + 1], op, value, path=target)
        edits.append(SetProjectValue(op=op, path=target, value=value))
    return StudioProjectFilePatch(address=PRESENTATION_ADDRESS, patches=edits)


def _model_files(
    folders: ProjectFolderService, snapshot: StudioProjectSnapshot,
    body: StudioDocumentAuthoringRequest, change: StudioModelAuthoringChange, document: str,
    keys: list[RunKey], lit: set[str], *, shared: bool, components: ComponentRegistry,
) -> tuple[list[StudioProjectFilePatch], str | None, str | None]:
    project = snapshot.project
    for step_id in lit:
        step = project.steps[step_id]
        if (
            not isinstance(step.base, InlineBase)
            or step.base.inline.get("graph") != {"document": document}
            or step.deltas or step.sweep is not None or (not shared and step.inputs)
        ):
            raise ProjectFolderInvalid("Topology edits require direct Model references without sweeps or composed inputs.")
    source = project.documents[document]
    if source.schema_id != "feedbax.spec.graph" or not isinstance(source.payload, dict):
        raise ProjectFolderInvalid("Choose a named Model document.")
    base = GraphSpec.model_validate(source.payload)
    plan = folders.read_plan(snapshot.project_id, revision=body.revision, source_runs=keys).plan
    resolved: dict[RunKey, tuple[dict[str, Any], GraphSpec, StudioTaskBindingSpec]] = {}
    reserved = set(base.nodes)
    for key in keys:
        definition = plan.selected_definitions.get(key)
        if definition is None or definition.document_ids.get("payload.graph") != document:
            if key.step in lit:
                raise ProjectFolderInvalid("A lit step no longer resolves this Model.")
            continue
        payload = definition.payload["graph"]
        if isinstance(payload, dict) and isinstance(nodes := payload.get("nodes"), dict):
            reserved.update(nodes)
        if key.step in lit:
            if not isinstance(payload, dict):
                raise ProjectFolderInvalid("A lit step has an invalid Model.")
            graph = GraphSpec.model_validate(payload)
            resolved[key] = (
                payload, graph, StudioTaskBindingSpec.model_validate(definition.payload.get("task_binding_spec") or {}),
            )
    if {key.step for key in resolved} != lit:
        raise ProjectFolderInvalid("A lit step no longer resolves this Model.")
    if not shared and len(resolved) != len(lit):
        raise ProjectFolderInvalid("Narrowed topology edits require one Model definition per lit step.")
    if snapshot.presentation is not None:
        shared_layout = snapshot.presentation.documents.get(document)
        if shared_layout is not None:
            reserved.update(shared_layout.positions)
        for presentation in snapshot.presentation.steps.values():
            reserved.update(presentation.model_positions.get(document, {}))

    def validate(payload: dict[str, Any], binding: StudioTaskBindingSpec) -> None:
        graph = GraphSpec.model_validate(payload)
        if isinstance(change.operation, StudioModelConnectPorts):
            validate_model_connections(graph, binding, components)
        elif isinstance(change.operation, StudioModelExposeOutput):
            node_id, port = change.operation.source_node, change.operation.source_port
            node = graph.nodes.get(node_id)
            if node is None or port not in node.output_ports:
                raise ProjectFolderInvalid(f"Public output {port!r} needs the Model port {node_id}.{port}.")
            if graph.output_ports.count(port) != 1 or graph.output_bindings.get(port) != (node_id, port):
                raise ProjectFolderInvalid(f"Public output {port!r} has a conflicting declaration.")

    files = []
    created = None
    if shared:
        model = prepare_model_edit(base, change, components, reserved)
        updated = deepcopy(source.payload)
        apply_document_edits(updated, model.step_edits())
        prior = {step: project.steps[step].document_overrides.get(document, {}) for step in sorted(lit)}
        differences = {step: model.supersede(edits, step) for step, edits in prior.items()}
        operation = change.operation
        for key, (_, effective, binding) in resolved.items():
            if isinstance(operation, StudioModelConnectPorts) and operation.wire.source_node == operation.wire.target_node:
                if prepare_model_edit(effective, change, components, reserved) != model:
                    raise ProjectFolderInvalid(f"{key.step} requires a different recurrent wire initializer.")
            payload = deepcopy(updated)
            apply_document_edits(payload, differences[key.step])
            validate(payload, binding)
        files.append(StudioProjectFilePatch(
            address=AuthoredFileAddress(kind="document", id=document), patches=model.document_edits(source.payload),
        ))
        files.extend(
            _step_difference(project, step, document, local)
            for step, local in differences.items() if local != prior[step]
        )
        created = model.component
    else:
        for key in sorted(resolved, key=lambda item: item.step):
            step = key.step
            raw, graph, binding = resolved[key]
            model = prepare_model_edit(graph, change, components, reserved)
            operations = model.step_edits()
            updated = deepcopy(raw)
            apply_document_edits(updated, operations)
            validate(updated, binding)
            prior = document_edit_ops(project.steps[step].document_overrides.get(document, {}))
            files.append(_step_difference(project, step, document, [*prior, *operations]))
            created = model.component
    presentation_revision = None
    if created and isinstance(change.operation, StudioModelAddComponent) and change.operation.position is not None:
        if body.presentation_revision is None:
            raise ProjectFolderInvalid("A component placement requires a presentation revision")
        if body.presentation_revision != snapshot.presentation_revision:
            raise ProjectRevisionConflict("presentation", snapshot.presentation_revision)
        files.append(_placed_component(
            snapshot, document, lit, shared=shared, component=created, operation=change.operation,
        ))
        presentation_revision = snapshot.presentation_revision
    return files, created, presentation_revision


def document_authoring(
    folders: ProjectFolderService, project_id: str, body: StudioDocumentAuthoringRequest,
    *, components: ComponentRegistry,
) -> StudioDocumentAuthoring:
    key = RunKey(body.run.variant, body.run.step, body.run.row)
    plan = folders.read_plan(project_id, revision=body.revision, source_runs=[key]).plan
    definition = plan.selected_definitions.get(key)
    prefix = f"/payload/{body.field}"
    document = definition.document_ids.get(f"payload.{body.field}") if definition is not None else None
    if definition is None or document is None or body.field not in definition.payload:
        raise ProjectFolderInvalid("The selected run has no named definition for this view.")
    sources = {
        path.removeprefix(prefix): StudioAuthoringSource(kind=origin.kind, name=origin.name, path=origin.path)
        for path, origin in definition.sources.items()
        if path.startswith(prefix + "/") and origin.kind in ("document", "step")
    }
    files = []
    if change := body.change:
        snapshot = folders.read(project_id, revision=body.revision)
        project = snapshot.project
        if project.sweep is not None:
            raise ProjectFolderInvalid("Edit the named definition outside project variants.")
        users = plan.document_users(key.variant, document)
        lit = set(change.lit_steps)
        if key.step not in lit or not lit <= users or len(lit) != len(change.lit_steps):
            raise ProjectFolderInvalid("The lit steps no longer match this document.")
        shared = lit == users
        relevant = [row for row in plan.rows if row.key.variant == key.variant and document in row.document_uses]
        if not shared and any(document not in row.document_ids.values() for row in relevant):
            raise ProjectFolderInvalid("This document is inherited through a resolved row and cannot be narrowed.")
        if isinstance(change, StudioModelAuthoringChange):
            if body.field != "graph":
                raise ProjectFolderInvalid("Topology edits require a Model.")
            try:
                files, created, presentation_revision = _model_files(
                    folders, snapshot, body, change, document, [row.key for row in relevant],
                    lit, shared=shared, components=components,
                )
            except ValueError as error:
                raise ProjectFolderInvalid(str(error)) from error
            return StudioDocumentAuthoring(
                revision=body.revision, document=document, payload=definition.payload[body.field],
                sources=sources, files=files, created_component=created, presentation_revision=presentation_revision,
            )
        source = sources.get(change.path)
        if source is not None and source.kind == "document" and source.name == document:
            path = source.path
        elif source is not None and source.kind == "step" and source.name == key.step:
            path = _step_source_path(project, key.step, document, source)
        else:
            raise ProjectFolderInvalid("This value is outside the named document.")
        if shared:
            files.append(StudioProjectFilePatch(
                address=AuthoredFileAddress(kind="document", id=document),
                patches=[SetProjectValue(
                    op="replace", path="/payload" + pointer(json_pointer_parts(path)), value=change.value,
                )],
            ))
        for step in sorted(lit):
            edit = _scalar_difference(project, step, document, path, change.value, shared=shared)
            if edit is not None:
                files.append(edit)
    return StudioDocumentAuthoring(
        revision=body.revision, document=document, payload=definition.payload[body.field],
        sources=sources, files=files,
    )
