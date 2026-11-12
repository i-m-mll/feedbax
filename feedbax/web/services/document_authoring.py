"""Prepare a revision-bound edit to a named document or its lit step users."""

from pydantic import JsonValue

from feedbax.component_registry import ComponentRegistry
from feedbax.contracts.graph import GraphSpec, StudioTaskBindingSpec
from feedbax.contracts.project import InlineBase, Project, json_pointer_parts
from feedbax.contracts.project_plan import RunKey
from feedbax.contracts.project_sources import pointer
from feedbax.contracts.studio_authoring import (
    StudioAuthoringSource,
    StudioDocumentAuthoring,
    StudioDocumentAuthoringRequest,
    StudioModelAuthoringChange,
)
from feedbax.contracts.studio_project import AuthoredFileAddress, SetProjectValue, StudioProjectFilePatch
from feedbax.web.services.model_authoring import prepare_model_edit
from feedbax.web.services.project_folder import ProjectFolderInvalid, ProjectFolderService


def _differences(
    project: Project, step: str, document: str, path: str, value: JsonValue, *, shared: bool,
) -> StudioProjectFilePatch | None:
    authored = project.steps[step]
    differences = {name: dict(edits) for name, edits in authored.document_overrides.items()}
    local = differences.setdefault(document, {})
    if shared:
        if path not in local:
            return None
        del local[path]
        if not local:
            del differences[document]
    else:
        local[path] = value
    return StudioProjectFilePatch(
        address=AuthoredFileAddress(kind="step", id=step),
        patches=[SetProjectValue(
            op="replace" if "document_overrides" in authored.model_fields_set else "add",
            path="/document_overrides", value=differences,
        )],
    )


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
        project = folders.read(project_id, revision=body.revision).project
        if project.sweep is not None:
            raise ProjectFolderInvalid("Edit the named definition outside project variants.")
        users = plan.document_users(key.variant, document)
        lit = set(change.lit_steps)
        if key.step not in lit or not lit <= users or len(lit) != len(change.lit_steps):
            raise ProjectFolderInvalid("The lit steps no longer match this document.")
        if isinstance(change, StudioModelAuthoringChange):
            if body.field != "graph" or lit != users:
                raise ProjectFolderInvalid("Topology edits require all Model users to be lit.")
            for step_id in lit:
                step = project.steps[step_id]
                if (
                    not isinstance(step.base, InlineBase)
                    or step.base.inline.get("graph") != {"document": document}
                    or step.document_overrides.get(document)
                    or step.deltas
                    or step.sweep is not None
                ):
                    raise ProjectFolderInvalid("Topology edits require direct Model references without step overrides or sweeps.")
            source = project.documents[document]
            if source.schema_id != "feedbax.spec.graph" or not isinstance(source.payload, dict):
                raise ProjectFolderInvalid("Choose a named Model document.")
            try:
                graph = GraphSpec.model_validate(source.payload)
                keys = [
                    row.key for row in plan.rows
                    if row.key.variant == key.variant and document in row.document_uses
                ]
                definitions = plan.selected_definitions
                if keys != [key]:
                    shared_plan = folders.read_plan(project_id, revision=body.revision, source_runs=keys).plan
                    definitions = shared_plan.selected_definitions
                bindings = []
                for run_key in keys:
                    resolved = definitions.get(run_key)
                    if (resolved is None or resolved.document_ids.get("payload.graph") != document
                            or GraphSpec.model_validate(resolved.payload.get("graph")) != graph):
                        raise ProjectFolderInvalid("Topology edits require the same named Model on every lit step.")
                    bindings.append(StudioTaskBindingSpec.model_validate(resolved.payload.get("task_binding_spec") or {}))
                patches, created = prepare_model_edit(source.payload, graph, change, components, bindings)
                files = [StudioProjectFilePatch(
                    address=AuthoredFileAddress(kind="document", id=document),
                    patches=patches,
                )]
            except ValueError as error:
                raise ProjectFolderInvalid(str(error)) from error
            return StudioDocumentAuthoring(
                revision=body.revision, document=document, payload=definition.payload[body.field],
                sources=sources, files=files, created_component=created,
            )
        source = sources.get(change.path)
        if source is not None and source.kind == "document" and source.name == document:
            path = source.path
        elif source is not None and source.kind == "step" and source.name == key.step:
            parts = json_pointer_parts(source.path)
            if len(parts) != 3 or parts[:2] != ("document_overrides", document):
                raise ProjectFolderInvalid("This value is outside the named document.")
            path = parts[2]
        else:
            raise ProjectFolderInvalid("This value is outside the named document.")
        shared = lit == users
        if not shared and any(
            row.key.variant == key.variant and document in row.document_uses
            and document not in row.document_ids.values()
            for row in plan.rows
        ):
            raise ProjectFolderInvalid("This document is inherited through a resolved row and cannot be narrowed.")
        if shared:
            files.append(StudioProjectFilePatch(
                address=AuthoredFileAddress(kind="document", id=document),
                patches=[SetProjectValue(
                    op="replace", path="/payload" + pointer(json_pointer_parts(path)), value=change.value,
                )],
            ))
        for step in sorted(lit):
            edit = _differences(project, step, document, path, change.value, shared=shared)
            if edit is not None:
                files.append(edit)
    return StudioDocumentAuthoring(
        revision=body.revision, document=document, payload=definition.payload[body.field],
        sources=sources, files=files,
    )
