"""Convert a saved Studio GraphProject into the managed text project format."""

from __future__ import annotations

import os
from pathlib import Path
import re
import sys

from pydantic import BaseModel, TypeAdapter, ValidationError

from feedbax.contracts.analysis_document import AnalysisDocument
from feedbax.contracts.canonical_json import canonical_json_v2_bytes
from feedbax.contracts.figure_document import PLOTLY_FIGURE_ROLE
from feedbax.contracts.graph import (
    STUDIO_GRAPH_PROJECT_SCHEMA_ID,
    STUDIO_GRAPH_PROJECT_SCHEMA_VERSION,
    STUDIO_WORKSPACE_SCHEMA_ID,
    STUDIO_WORKSPACE_SCHEMA_VERSION,
    WORKSPACE_DOCUMENT_SCHEMA_ID,
    WORKSPACE_DOCUMENT_SCHEMA_VERSION,
    AnalysisPageSpec,
    GraphProject,
    StudioScenarioSpec,
    StudioStageSpec,
)
from feedbax.contracts.migrations import default_spec_registry
from feedbax.contracts.project import Document, DocumentReference, InlineBase, Payload, Project, Step, StepKind
from feedbax.contracts.project_inputs import InputBinding, NamedSubset, ProjectName, RunReference, StepReference
from feedbax.contracts.project_io import ProjectConfiguration
from feedbax.contracts.project_sources import pointer
from feedbax.contracts.report_document import ReportDocument
from feedbax.contracts.request_specs import EvaluationRequestSpec
from feedbax.contracts.rollout import STUDIO_DEFAULT_EVALUATION_TYPE
from feedbax.contracts.selection import SelectionSpec
from feedbax.contracts.strict_json import strict_json_loads
from feedbax.contracts.studio_project import DocumentPresentation, PresentationCamera, PresentationDocument, PresentationPosition
from feedbax.contracts.studio_training import STUDIO_GRAPH_METHOD
from feedbax.contracts.training import ObjectiveSlotSpec, TaskSpec, TrainingSpec
from feedbax.studio.protocol import infer_task_n_steps
from feedbax.studio.training_requests import StudioTrainingLoweringError, studio_graph_training_settings
from feedbax.project_folder import write_new_project_folder


_STAGE_KINDS: dict[str, StepKind] = {
    "train": "train",
    "eval": "evaluate",
    "analysis": "analyze",
    "report": "report",
}
_NAME = TypeAdapter(ProjectName)
_STUDIO_OBJECTIVE_VERSION = default_spec_registry.resolve("StudioObjectiveSpec").current_version

# The saved Studio v1 objective projects these three legacy probe names into
# graph port substates. Keep this reader fixed to that saved wire format.
_LEGACY_PROBE_SUBSTATES = {
    "probe:effector_pos": ("mechanics", "effector", "position", "states.mechanics.effector.pos"),
    "probe:effector_vel": ("mechanics", "effector", "velocity", "states.mechanics.effector.vel"),
    "probe:network_hidden": ("network", "hidden", "hidden", "states.net.hidden"),
}


class WorkspaceMigrationError(ValueError):
    def __init__(self, path: str, message: str) -> None:
        super().__init__(f"{path or '/'}: {message}")


def _validate[Model: BaseModel](model: type[Model], value: object, path: str) -> Model:
    try:
        return model.model_validate(value)
    except ValidationError as error:
        issue = error.errors(include_url=False, include_input=False)[0]
        raise WorkspaceMigrationError(
            path + pointer(tuple(map(str, issue["loc"]))), issue["msg"],
        ) from error


def _without_studio_stamp(value: Payload, kind: str) -> Payload:
    """Remove registry metadata supplied by Studio save admission."""
    family = default_spec_registry.resolve(kind)
    stamp = {"schema_id": family.identity, "schema_version": family.current_version}
    return {field: item for field, item in value.items() if field not in stamp or item != stamp[field]}


def _read_source(source: Path) -> GraphProject:
    if not source.is_file():
        raise WorkspaceMigrationError("", "Expected a saved Studio workspace JSON file.")
    raw = strict_json_loads(source.read_bytes(), ref=str(source))
    canonical_json_v2_bytes(raw)
    if not isinstance(raw, dict):
        raise WorkspaceMigrationError("", "Expected a saved Studio workspace object.")
    formats = (
        ("", raw, STUDIO_GRAPH_PROJECT_SCHEMA_ID, STUDIO_GRAPH_PROJECT_SCHEMA_VERSION),
        ("/workspace", raw.get("workspace"), STUDIO_WORKSPACE_SCHEMA_ID, STUDIO_WORKSPACE_SCHEMA_VERSION),
        ("/workspace_document", raw.get("workspace_document"), WORKSPACE_DOCUMENT_SCHEMA_ID,
         WORKSPACE_DOCUMENT_SCHEMA_VERSION),
    )
    for path, value, schema_id, version in formats:
        if not isinstance(value, dict):
            raise WorkspaceMigrationError(path, "Expected the object written by the Studio workspace writer.")
        for field, expected in (("schema_id", schema_id), ("schema_version", version)):
            if value.get(field) != expected:
                raise WorkspaceMigrationError(f"{path}/{field}", f"Expected {expected!r} from the saved Studio writer.")
    return _validate(GraphProject, raw, "")


def _name(value: str, default: str) -> str:
    return re.sub(r"[^A-Za-z0-9_-]+", "-", value).strip("-") or default


def _unique_name(value: str, used: set[str]) -> str:
    name = value
    suffix = 2
    while name.casefold() in used:
        name = f"{value}-{suffix}"
        suffix += 1
    used.add(name.casefold())
    return name


def _legacy_selector(selector: str | None) -> Payload | None:
    if not selector:
        return None
    if selector in _LEGACY_PROBE_SUBSTATES:
        node, port, subpath, state_path = _LEGACY_PROBE_SUBSTATES[selector]
        return {
            "namespace": "state_path", "compact": f"path:{state_path}", "target_id": node,
            "path": state_path, "role": "observed", "metadata": {
                "source": "port_substate_selector", "legacy_selector": selector, "direction": "output",
                "subpath": subpath, "graph_port_node_id": node, "graph_port_name": port,
                "graph_port_direction": "output", "graph_port_compact": f"port:{node}.{port}",
            },
        }
    prefix, separator, value = selector.partition(":")
    if not separator:
        prefix = ""
    if prefix == "probe":
        namespace, target, path = "probe", value, None
    elif prefix == "port":
        node, _, port = value.partition(".")
        namespace, target, path = "graph_port", node or None, port or None
    elif prefix == "path":
        namespace, target, path = "state_path", None, value
    else:
        namespace, target, path = "custom", None, selector
    return {
        "namespace": namespace, "compact": selector, "target_id": target, "path": path,
        "metadata": {"source": "legacy_loss_selector"},
    }


def _legacy_objective_terms(loss: Payload, path: tuple[str, ...] = ()) -> list[Payload]:
    children = loss.get("children")
    if children:
        return [
            term
            for key, child in children.items()
            for term in _legacy_objective_terms(child, (*path, key))
        ]
    kind = loss["type"]
    term: Payload = {
        "id": "objective:" + (".".join(path) if path else "root"),
        "type_id": kind,
        "label": loss["label"],
        "role": "regularizer" if "regular" in kind.lower() else "loss",
        "source_selector": _legacy_selector(loss.get("selector")),
        "target_selector": None,
        "operator": "minimize",
        "penalty": loss.get("norm"),
        "temporal_selector": loss.get("time_agg"),
        "metadata": {"legacy_loss_path": list(path), "legacy_loss_type": kind},
    }
    if "weight" in loss:
        term["weight"] = loss["weight"]
    return [term]


def _check_objective_mirror(objective: Payload, loss: Payload | None, path: str) -> None:
    """Accept the saved editor projection only when the legacy loss contains all its authoring."""
    expected = {
        "schema_version": _STUDIO_OBJECTIVE_VERSION,
        "legacy_loss_spec": loss,
        "metadata": {"lowered_from": "training_spec.loss"},
    }
    for field, value in expected.items():
        if field not in objective or canonical_json_v2_bytes(objective[field]) != canonical_json_v2_bytes(value):
            raise WorkspaceMigrationError(f"{path}/{field}", "Expected the Studio projection of the saved training loss.")
    for field in objective.keys() - expected.keys() - {"terms"}:
        raise WorkspaceMigrationError(path + pointer((field,)), "No corresponding Project objective field.")
    if loss is None or canonical_json_v2_bytes(objective.get("terms")) != canonical_json_v2_bytes(_legacy_objective_terms(loss)):
        raise WorkspaceMigrationError(f"{path}/terms", "The authored objective terms differ from the saved training loss.")


def _train_payload(
    scenario: StudioScenarioSpec, path: str, documents: dict[str, Document],
    tasks: dict[bytes, str],
) -> Payload:
    payload: Payload = {"graph": {"document": "model"}}
    task = None
    if scenario.task_spec is not None:
        task_payload = _without_studio_stamp(scenario.task_spec, "TaskSpec")
        for field in task_payload.keys() - TaskSpec.model_fields.keys():
            raise WorkspaceMigrationError(path + pointer(("task_spec", field)), "No corresponding Project Task field.")
        task = _validate(TaskSpec, task_payload, f"{path}/task_spec")
        data = task.model_dump(mode="json", exclude_none=True)
        key = canonical_json_v2_bytes(data)
        if key not in tasks:
            name = "task" if not tasks else f"task-{len(tasks) + 1}"
            tasks[key] = name
            documents[name] = Document(
                schema=default_spec_registry.resolve(TaskSpec.__name__).identity, payload=data,
            )
        payload["task"] = {"document": tasks[key]}
    if scenario.task_binding_spec is not None:
        payload["task_binding_spec"] = scenario.task_binding_spec.model_dump(mode="json", exclude_none=True)

    raw = _without_studio_stamp(scenario.training_spec, "TrainingSpec") if scenario.training_spec else None
    settings = _validate(TrainingSpec, raw, f"{path}/training_spec") if raw else None
    objective = scenario.objective_spec
    if objective:
        if objective.get("schema_version") == _STUDIO_OBJECTIVE_VERSION:
            _check_objective_mirror(objective, raw["loss"] if settings is not None else None, f"{path}/objective_spec")
        else:
            slot = _validate(ObjectiveSlotSpec, objective, f"{path}/objective_spec")
            if raw is not None:
                legacy = _validate(ObjectiveSlotSpec, {"loss": raw.get("loss")}, f"{path}/training_spec")
                if slot != legacy:
                    raise WorkspaceMigrationError(f"{path}/objective_spec", "The authored objective differs from the training loss.")
            payload["objective"] = slot.model_dump(mode="json", exclude_unset=True)
    if settings is None:
        return payload

    for field in raw:
        if field not in TrainingSpec.model_fields and field != "seed":
            raise WorkspaceMigrationError(path + pointer(("training_spec", field)), "No corresponding Project Train field.")
    for field in ("n_epochs", "early_stopping"):
        if getattr(settings, field) is not None:
            raise WorkspaceMigrationError(f"{path}/training_spec/{field}", "No corresponding Project Train field.")
    try:
        authored = studio_graph_training_settings(settings, raw, infer_task_n_steps(task) if task is not None else None)
    except StudioTrainingLoweringError as error:
        raise WorkspaceMigrationError(path + error.path, error.message) from error
    return payload | {
        "training_config": authored.config.model_dump(mode="json", exclude_unset=True),
        "objective": authored.objective.model_dump(mode="json", exclude_unset=True),
        "method_ref": STUDIO_GRAPH_METHOD.model_dump(mode="json"),
        "method_payload": authored.method_payload.model_dump(mode="json"),
    }


def _historical_inputs(
    stage: StudioStageSpec, path: str, notices: list[str],
) -> dict[str, InputBinding | None]:
    source = stage.selection_spec
    if stage.kind == "eval":
        role, ids_key = "subject", "training_run_ids"
    elif stage.kind == "analysis":
        role, ids_key = "states", "eval_run_ids"
    else:
        return {}
    ids = source.get(ids_key, [])
    at = f"{path}/selection_spec/{ids_key}"
    if stage.kind == "eval":
        for name in ("selection_spec", "training_selection"):
            if name in source and source[name] is not None:
                select = _validate(SelectionSpec, source[name], f"{path}/selection_spec/{name}")
                if select.mode != "explicit" or select.manifest_kind != "TrainingRunManifest":
                    raise WorkspaceMigrationError(
                        f"{path}/selection_spec/{name}", "This historical selection has no Project input binding.",
                    )
                if ids and ids != select.ids:
                    raise WorkspaceMigrationError(at, f"The selected IDs differ from {name}/ids.")
                ids, at = select.ids, f"{path}/selection_spec/{name}/ids"
                break
    if not isinstance(ids, list):
        raise WorkspaceMigrationError(at, "Expected a list of saved run IDs.")
    references = []
    for index, run_id in enumerate(ids):
        if not isinstance(run_id, str) or not run_id or run_id.strip() != run_id:
            raise WorkspaceMigrationError(f"{at}/{index}", "Expected a nonempty run ID without surrounding whitespace.")
        reference = RunReference(run=run_id)
        references.append(reference)
        notices.append(f"{at}/{index}: kept historical run {reference.run!r}; Store migration is required to resolve it.")
    if not references:
        return {}
    return {role: references[0] if len(references) == 1 else references}


def _step_inputs(
    stage: StudioStageSpec, path: str, stage_names: dict[str, str], notices: list[str],
) -> dict[str, InputBinding | None]:
    historical = _historical_inputs(stage, path, notices)
    raw = stage.selection_spec.get("inputs")
    if raw is None:
        return historical
    at = f"{path}/selection_spec/inputs"
    if not isinstance(raw, dict):
        raise WorkspaceMigrationError(at, "Expected named input bindings.")
    if historical.keys() & raw.keys():
        raise WorkspaceMigrationError(at, "A role also has a historical run selection.")
    bindings = dict(historical)
    for role, value in raw.items():
        role_path = at + pointer((role,))
        try:
            _NAME.validate_python(role)
        except ValidationError as error:
            raise WorkspaceMigrationError(role_path, "Invalid Project input role.") from error
        if not isinstance(value, dict) or "step" not in value:
            raise WorkspaceMigrationError(role_path, "Expected a reference to a saved workspace stage.")
        step = value["step"]
        if not isinstance(step, str) or step not in stage_names and step not in stage_names.values():
            raise WorkspaceMigrationError(f"{role_path}/step", f"No migrated stage matches {step!r}.")
        bindings[role] = _validate(
            StepReference, value | {"step": stage_names.get(step, step)}, role_path,
        )
    return bindings


def _page_authoring(page: AnalysisPageSpec, layout_nodes: set[str] | None = None) -> str | None:
    graph = page.graph_spec
    nodes = graph.get("nodes") or {}
    if not isinstance(nodes, dict) or any(
        name not in (layout_nodes or ()) or node != {} for name, node in nodes.items()
    ) or graph.get("wires") or graph.keys() - {"nodes", "wires", "dataSourceId", "data_source_id"}:
        return "graph_spec"
    return next((field for field in ("eval_params", "input_requirements", "eval_run_id") if getattr(page, field)), None)


def _analysis_nodes(analysis: AnalysisDocument | None) -> set[str]:
    nodes = {"states"}
    if analysis is not None:
        nodes.update(analysis.operations)
        nodes.update(analysis.views)
    return nodes


def _analysis_payload(raw: dict, path: str) -> tuple[Document, AnalysisDocument | None]:
    raw = _without_studio_stamp(raw, "AnalysisRunSpec")
    if "pages" not in raw:
        analysis = _validate(AnalysisDocument, raw, path)
        for element, view in analysis.views.items():
            if isinstance(view.figure, DocumentReference):
                raise WorkspaceMigrationError(
                    path + pointer(("views", element, "figure", "document")),
                    "The saved workspace does not contain that named Figure document.",
                )
        return Document(schema=analysis.schema_version, payload=analysis.model_dump(mode="json", exclude_unset=True)), analysis

    pages = raw["pages"]
    if not isinstance(pages, list):
        raise WorkspaceMigrationError(f"{path}/pages", "Expected saved Analysis pages.")
    for index, value in enumerate(pages):
        at = f"{path}/pages/{index}"
        page = _validate(AnalysisPageSpec, value, at)
        if field := _page_authoring(page):
            raise WorkspaceMigrationError(f"{at}/{field}", "The legacy canvas has no typed Project Analyze equivalent.")
    ignored = {"pages", "schema_version", "active_page_id", "metadata", "input_collections"}
    for field, value in raw.items():
        if field not in ignored and value and (field != "evaluation_states_policy" or value != "recompute"):
            raise WorkspaceMigrationError(path + pointer((field,)), "No Project Analyze field preserves this authoring.")
    schema = AnalysisDocument.model_fields["schema_version"].default
    return Document(schema=schema, payload={"schema_version": schema, "operations": {}, "views": {}}), None


def _presentation(
    source: GraphProject, document_ids: dict[str, str],
    analyses: dict[str, AnalysisDocument | None], notices: list[str],
) -> PresentationDocument:
    workspace = source.workspace_document
    ui = workspace.graph_ui_state
    model_path = "/workspace_document/graph_ui_state"
    positions = {}
    for node, state in ui.node_states.items():
        if node in source.graph.nodes:
            positions[node] = _validate(
                PresentationPosition, state.position.model_dump(),
                model_path + pointer(("node_states", node, "position")),
            )
        else:
            notices.append(model_path + pointer(("node_states", node)) + ": no top-level Model node matches this position.")
    for key in ("edge_states", "subgraph_states", "tap_states"):
        if getattr(ui, key):
            notices.append(f"{model_path}/{key}: this canvas layout has no Project presentation address.")
    camera = _validate(PresentationCamera, ui.viewport.model_dump(), f"{model_path}/viewport")
    layout = {"model": DocumentPresentation(positions=positions, camera=camera)}
    pages = {page.id: page for page in workspace.analysis_pages}
    for stage, stage_layout in workspace.analysis_canvas_layout.stages.items():
        path = "/workspace_document/analysis_canvas_layout/stages" + pointer((stage,))
        if stage not in analyses:
            notices.append(f"{path}: no converted Analysis document owns this layout.")
            continue
        if len(stage_layout.pages) != 1:
            notices.append(f"{path}/pages: a Project Analysis document has one canvas.")
            continue
        (page, view), = stage_layout.pages.items()
        path += pointer(("pages", page))
        analysis = analyses[stage]
        names = _analysis_nodes(analysis)
        graph = pages[page].graph_spec
        data_source = graph.get("dataSourceId", graph.get("data_source_id"))
        view_positions = {}
        for node, position in view.node_positions.items():
            target = "states" if node == data_source else node
            if target in names:
                view_positions[target] = PresentationPosition(x=position.x, y=position.y)
            else:
                notices.append(path + pointer(("node_positions", node)) + ": no converted Analysis element matches this position.")
        layout[document_ids[stage]] = DocumentPresentation(
            positions=view_positions, camera=PresentationCamera.model_validate(view.viewport.model_dump()),
        )
    return PresentationDocument(documents=layout)


def convert_workspace(source: GraphProject) -> tuple[Project, PresentationDocument, list[str]]:
    workspace = source.workspace
    if workspace is None:
        raise WorkspaceMigrationError("/workspace", "Expected the saved semantic workspace.")
    notices: list[str] = []
    ids: dict[str, str] = {}
    used: set[str] = set()
    for index, stage in enumerate(workspace.stages):
        path = f"/workspace/stages/{index}"
        if stage.scenario_id is None:
            continue
        if stage.id in ids:
            raise WorkspaceMigrationError(f"{path}/id", f"Stage {stage.id!r} is repeated.")
        if stage.kind not in _STAGE_KINDS:
            raise WorkspaceMigrationError(f"{path}/kind", f"No Project step represents {stage.kind!r}.")
        ids[stage.id] = _unique_name(_name(stage.id.removeprefix("stage:"), stage.kind), used)

    documents = {"model": Document(schema=source.graph.schema_id, payload=source.graph.model_dump(mode="json"))}
    steps: dict[str, Step] = {}
    tasks: dict[bytes, str] = {}
    analyses: dict[str, AnalysisDocument | None] = {}
    analysis_documents: dict[str, str] = {}
    document_names = {"model", "task"}
    for index, stage in enumerate(workspace.stages):
        if stage.scenario_id is None:
            continue
        at = f"/workspace/stages/{index}"
        scenario = workspace.scenarios.get(stage.scenario_id)
        if scenario is None:
            raise WorkspaceMigrationError(f"{at}/scenario_id", f"No saved scenario matches {stage.scenario_id!r}.")
        path = "/workspace/scenarios" + pointer((stage.scenario_id,))
        if scenario.id != stage.scenario_id:
            raise WorkspaceMigrationError(f"{path}/id", "Scenario ID differs from its saved key.")
        for key in ("matrix", "sweep_matrix", "run_matrix", "condition_matrix"):
            if stage.selection_spec.get(key):
                raise WorkspaceMigrationError(f"{at}/selection_spec/{key}", "This workspace matrix has no Project sweep conversion.")
        kind = _STAGE_KINDS[stage.kind]
        kept = {
            "train": {"training_spec", "task_spec", "task_binding_spec", "objective_spec"},
            "evaluate": set(),
            "analyze": {"analysis_spec"},
            "report": {"report_spec"},
        }[kind]
        for field in (
            "training_spec", "task_spec", "task_binding_spec", "objective_spec",
            "analysis_spec", "report_spec", "probe_specs", "temporal_spec", "biomechanics_spec",
        ):
            if field not in kept and getattr(scenario, field):
                raise WorkspaceMigrationError(f"{path}/{field}", "No Project step field preserves this scenario authoring.")
        if scenario.parent_scenario_id is not None:
            parent = workspace.scenarios.get(scenario.parent_scenario_id)
            if parent is None:
                raise WorkspaceMigrationError(f"{path}/parent_scenario_id", "The parent scenario is missing.")
            if kind != "evaluate" or not any(
                candidate.kind == "train" and candidate.scenario_id == parent.id for candidate in workspace.stages
            ):
                raise WorkspaceMigrationError(f"{path}/parent_scenario_id", "No Project step base preserves this scenario inheritance.")

        name = ids[stage.id]
        inputs = _step_inputs(stage, at, ids, notices)
        base: InlineBase | DocumentReference = InlineBase(inline={})
        if kind == "train":
            base = InlineBase(inline=_train_payload(scenario, path, documents, tasks))
        elif kind == "evaluate":
            params = stage.selection_spec.get("eval_params")
            if params is not None and not isinstance(params, dict):
                raise WorkspaceMigrationError(f"{at}/selection_spec/eval_params", "Expected authored Evaluate parameters.")
            if params is not None or inputs:
                spec = EvaluationRequestSpec(evaluation_type=STUDIO_DEFAULT_EVALUATION_TYPE, params=params or {})
                base = InlineBase(inline=spec.model_dump(mode="json", exclude_unset=True))
        elif kind == "analyze" and scenario.analysis_spec:
            doc = _unique_name(f"{name}-analysis", document_names)
            documents[doc], analyses[stage.id] = _analysis_payload(scenario.analysis_spec, f"{path}/analysis_spec")
            analysis_documents[stage.id] = doc
            base = DocumentReference(document=doc)
        elif kind == "report" and scenario.report_spec:
            raw = _without_studio_stamp(scenario.report_spec, "ReportSpec")
            if "report_type" in raw:
                raise WorkspaceMigrationError(f"{path}/report_spec/report_type", "The legacy report recipe has no Project document equivalent.")
            report = _validate(ReportDocument, raw, f"{path}/report_spec")
            for role in report.figure_roles:
                binding = inputs.get(role)
                if not isinstance(binding, StepReference) or binding.product != PLOTLY_FIGURE_ROLE:
                    index = next(index for index, block in enumerate(report.blocks) if getattr(block, "figure", None) == role)
                    raise WorkspaceMigrationError(f"{path}/report_spec/blocks/{index}/figure", "No exact Analyze figure product is bound to this role.")
            doc = _unique_name(f"{name}-report", document_names)
            documents[doc] = Document(schema=report.schema_version, payload=report.model_dump(mode="json", exclude_unset=True))
            base = DocumentReference(document=doc)
        steps[name] = Step(kind=kind, base=base, inputs=inputs)

    analysis_by_step = {ids[stage]: document for stage, document in analyses.items()}
    for index, stage in enumerate(workspace.stages):
        if stage.id not in ids:
            continue
        for role, binding in steps[ids[stage.id]].inputs.items():
            if not isinstance(binding, StepReference):
                continue
            at = f"/workspace/stages/{index}/selection_spec/inputs" + pointer((role,))
            if isinstance(binding.where, NamedSubset):
                raise WorkspaceMigrationError(f"{at}/where/subset", "The workspace has no named Project subset document.")
            if binding.element is not None:
                analysis = analysis_by_step.get(binding.step)
                if analysis is None or binding.element not in analysis.views:
                    raise WorkspaceMigrationError(f"{at}/element", "No converted Analysis figure matches this reference.")

    layout_nodes: dict[str, set[str]] = {}
    for stage, stage_layout in source.workspace_document.analysis_canvas_layout.stages.items():
        if stage in analyses:
            nodes = _analysis_nodes(analyses[stage])
            for page in stage_layout.pages:
                layout_nodes[page] = layout_nodes.get(page, nodes) & nodes
    for index, page in enumerate(source.workspace_document.analysis_pages):
        if field := _page_authoring(page, layout_nodes.get(page.id)):
            raise WorkspaceMigrationError(
                f"/workspace_document/analysis_pages/{index}/{field}",
                "No converted Analysis document preserves this authored page.",
            )

    return Project(documents=documents, steps=steps), _presentation(
        source, analysis_documents, analyses, notices,
    ), notices


def migrate_project(source: Path, out: Path | None) -> int:
    source = source.expanduser().absolute()
    destination = out.expanduser().absolute() if out is not None else source.with_name(_name(source.stem, "project"))
    try:
        if os.path.lexists(destination):
            raise FileExistsError(f"Destination already exists: {destination}")
        configuration = _validate(
            ProjectConfiguration, {"project": destination.name, "store": "results"}, "/destination",
        )
        project, presentation, notices = convert_workspace(_read_source(source))
        write_new_project_folder(destination, project, configuration, presentation)
    except (OSError, ValueError, RecursionError) as error:
        print(f"feedbax migrate-project: {source}: {error}", file=sys.stderr)
        return 2
    for notice in notices:
        print(f"{source}:{notice}", file=sys.stderr)
    print(destination)
    return 0
