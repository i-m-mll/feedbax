from __future__ import annotations

import sqlite3
import queue
import threading
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from feedbax.contracts.manifest import (
    AnalysisRunManifest,
    AnalysisRunSpec,
    ArrayStoreRef,
    ArtifactRef,
    EntrypointRef,
    EvaluationRunManifest,
    EvaluationRunSpec,
    ModelArtifactManifest,
    ParentRef,
    Provenance,
    ReportManifest,
    ReportSpec,
    TrainingRunManifest,
    load_manifest,
    sha256_file,
    spec_payload,
    write_manifest,
    write_training_run_manifest,
)
from feedbax.persistence.manifest_index import rebuild_manifest_index
from feedbax.integrations.provider import (
    component_registry_snapshot,
    provider_manifest,
    validate_analysis_spec,
    validate_evaluation_spec,
    validate_graph_spec,
    validate_report_spec,
    validate_spec,
    validate_task_spec,
    validate_training_spec,
)
from feedbax.studio.schema import (
    RuntimeIntrospectionResult,
    RuntimeSampleLeafSchema,
    enumerate_studio_schema_registry,
    validate_graph_connection_schema,
    validate_task_binding_schema,
)
from feedbax.contracts.graphs.normalization import normalize_graph_for_studio_authoring
from feedbax.web.app import create_app
from feedbax.contracts.graph import (
    GRAPH_SPEC_SCHEMA_ID,
    GRAPH_SPEC_SCHEMA_VERSION,
    AnalysisInputRequirement,
    ComponentSpec,
    GraphMetadata,
    GraphSpec,
    StudioStageSpec,
    StudioTaskBindingSpec,
    StudioTaskTimelineSpec,
    TapSpec,
    build_default_studio_workspace,
)
from feedbax.contracts.training import LossTermSpec, TaskSpec, TrainingRunSpec, TrainingSpec
from feedbax.contracts.migrations import default_spec_registry
from feedbax.studio.protocol import infer_task_n_steps
from feedbax.web.worker.app import (
    WorkerStatus,
    _Job,
    _extract_training_cfg,
    _require_worker_specs,
    _run_training,
    _write_job_manifest,
)

pytestmark = [pytest.mark.feedbax_contract, pytest.mark.provider_contract]


def _minimal_graph_spec() -> dict:
    return {
        "nodes": {
            "gain": {
                "type": "Gain",
                "params": {"gain": 2.0},
                "input_ports": ["input"],
                "output_ports": ["output"],
            }
        },
        "wires": [],
        "input_ports": [],
        "output_ports": ["output"],
        "input_bindings": {},
        "output_bindings": {"output": ("gain", "output")},
    }


def _runtime_network_graph_spec() -> dict:
    return {
        "nodes": {
            "network": {
                "type": "SimpleStagedNetwork",
                "params": {
                    "hidden_size": 100,
                    "input_size": 4,
                    "output_size": 2,
                    "hidden_type": "GRUCell",
                    "out_nonlinearity": "tanh",
                },
                "input_ports": ["target"],
                "output_ports": ["output", "hidden"],
            },
            "mechanics": {
                "type": "PointMass",
                "params": {"dt": 0.02},
                "input_ports": ["force"],
                "output_ports": ["effector"],
            },
        },
        "wires": [
            {
                "source_node": "network",
                "source_port": "output",
                "target_node": "mechanics",
                "target_port": "force",
            }
        ],
        "input_ports": ["target"],
        "output_ports": ["effector"],
        "input_bindings": {"target": ("network", "target")},
        "output_bindings": {"effector": ("mechanics", "effector")},
    }


def _task_input_binding(
    *,
    target_node_id: str = "network",
    target_port: str = "input",
    expected_shape: list | None = None,
) -> dict:
    data = {
        "id": "inputs",
        "label": "Inputs",
        "kind": "signal",
        "role": "model_input",
        "path": "inputs",
        "bindable": True,
        "dtype": "vector",
        "metadata": {},
    }
    if expected_shape is not None:
        data["expected_shape"] = expected_shape
    return {
        "schema_version": "feedbax.spec.studio.task_bindings.v2",
        "exposed_data": [data],
        "bindings": [
            {
                "id": f"task:inputs->{target_node_id}:{target_port}",
                "source_data_id": "inputs",
                "target_node_id": target_node_id,
                "target_port": target_port,
                "role": "model_input",
                "metadata": {},
            }
        ],
        "metadata": {},
    }


def _minimal_training_spec() -> dict:
    return {
        "optimizer": {"type": "adamw", "params": {"learning_rate": 0.001}},
        "loss": {
            "type": "composite",
            "label": "total",
            "weight": 1.0,
            "children": {
                "tracking": {
                    "type": "target",
                    "label": "tracking",
                    "weight": 1.0,
                    "selector": "graph_output:output",
                    "target_value": 0.0,
                }
            },
        },
        "n_batches": 2,
        "batch_size": 4,
    }


def _schema_workspace():
    graph = GraphSpec(
        nodes={
            "network": {
                "type": "Gain",
                "params": {"gain": 1.0},
                "input_ports": ["input"],
                "output_ports": ["output"],
            }
        },
        output_ports=["model_output"],
        output_bindings={"model_output": ("network", "output")},
        taps=[
            {
                "id": "activation-tap",
                "type": "probe",
                "position": {"afterNode": "network"},
                "paths": {"hidden": "state.network.hidden"},
            }
        ],
        metadata=GraphMetadata(
            name="Schema provider smoke",
            created_at="2026-05-20T00:00:00+00:00",
            updated_at="2026-05-20T00:00:00+00:00",
        ),
    )
    workspace = build_default_studio_workspace(label="Schema provider", graph=graph)
    train_stage = next(stage for stage in workspace.stages if stage.kind == "train")
    scenario = workspace.scenarios[train_stage.scenario_id]
    scenario.task_binding_spec = StudioTaskBindingSpec.model_validate(
        {
            "schema_version": "feedbax.spec.studio.task_bindings.v2",
            "exposed_data": [
                {
                    "id": "inputs",
                    "label": "Inputs",
                    "kind": "signal",
                    "path": "inputs",
                    "bindable": True,
                    "dtype": "vector",
                    "metadata": {},
                },
                {
                    "id": "targets",
                    "label": "Targets",
                    "kind": "target",
                    "path": "targets",
                    "bindable": False,
                    "dtype": "state",
                    "metadata": {},
                },
            ],
            "bindings": [
                {
                    "id": "task:inputs->network:input",
                    "source_data_id": "inputs",
                    "target_node_id": "network",
                    "target_port": "input",
                    "role": "model_input",
                    "metadata": {},
                }
            ],
            "metadata": {},
        }
    )
    scenario.objective_spec = {
        "terms": [{"selector": "task_data:targets", "label": "Target tracking"}]
    }
    scenario.probe_specs = [{"id": "manual-probe", "label": "Manual probe"}]
    workspace.stages.append(
        StudioStageSpec(
            id="stage:broken",
            kind="eval",
            label="Broken",
            scenario_id="scenario:missing",
        )
    )
    return workspace


def test_provider_manifest_exposes_phase_one_capabilities() -> None:
    manifest = provider_manifest()

    assert manifest.provider == "feedbax"
    assert manifest.capabilities["validate_graph_spec"].input_schema == "GraphSpec"
    assert manifest.capabilities["validate_training_spec"].input_schema == "TrainingRunSpec"
    assert manifest.capabilities["start_training_run"].input_schema == "TrainingRunSpec"
    assert manifest.capabilities["start_training_run"].output_schema == "TrainingRunManifest"
    assert manifest.capabilities["start_training_run"].action == "execute"
    assert manifest.capabilities["start_training_run"].requires_review
    assert manifest.capabilities["start_training_run"].mutates_state
    assert manifest.capabilities["start_training_run"].may_launch_compute
    assert "training_checkpoint" in manifest.capabilities["start_training_run"].artifact_roles
    assert manifest.capabilities["enumerate_studio_schemas"].output_schema == "StudioSchemaRegistry"
    assert "training_checkpoint" in manifest.artifact_roles
    assert "model_parameters" in manifest.artifact_roles
    assert "array_store" in manifest.artifact_roles
    assert "TrainingRunManifest" in manifest.schemas
    assert "TrainingRunSpec" in manifest.schemas
    assert "ModelArtifactManifest" in manifest.schemas
    assert "CheckpointSelectionManifest" in manifest.schemas
    assert "CheckpointSelectionSpec" in manifest.schemas
    assert "ArrayStorePayload" in manifest.schemas
    assert "ArrayStoreRef" in manifest.schemas
    assert "ObjectiveSpec" in manifest.schemas
    assert "AnalysisInputRequirement" in manifest.schemas
    assert "StudioSchemaRegistry" in manifest.schemas
    assert "TaskDataSchema" in manifest.schemas
    assert "RuntimeIntrospectionOptions" in manifest.schemas
    assert "RuntimeSampleLeafSchema" in manifest.schemas
    assert "MandibleManifestMapping" in manifest.schemas
    assert "ExecutionPlan" in manifest.schemas
    assert "LocalExecutionResult" in manifest.schemas


def test_provider_manifest_exports_governed_execution_artifact_refs() -> None:
    manifest = provider_manifest()
    plan_schema = manifest.schemas["ExecutionPlan"]
    result_schema = manifest.schemas["LocalExecutionResult"]

    assert (
        plan_schema["properties"]["artifact_routes"]["items"]["$ref"]
        == "#/$defs/ArtifactRef"
    )
    for field in ("stdout", "stderr", "manifest", "execution_plan"):
        assert result_schema["properties"][field]["$ref"] == "#/$defs/ArtifactRef"
    assert (
        result_schema["properties"]["produced_artifacts"]["items"]["$ref"]
        == "#/$defs/ArtifactRef"
    )

    prepare_roles = set(manifest.capabilities["prepare_execution_plan"].artifact_roles)
    assert {
        "execution_plan",
        "execution_log",
        "training_run_spec",
        "training_run_manifest",
        "tracked_spec",
        "bulk_output",
    }.issubset(prepare_roles)
    local_roles = set(manifest.capabilities["run_local_execution"].artifact_roles)
    assert {
        "execution_plan",
        "execution_log",
        "execution_stdout",
        "execution_stderr",
        "training_run_manifest",
    }.issubset(local_roles)


def test_provider_manifest_exposes_eval_analysis_report_action_depth() -> None:
    manifest = provider_manifest()

    expected = {
        "execute_evaluation_run": ("EvaluationRunSpec", "EvaluationRunManifest", "execute"),
        "execute_analysis_run": ("AnalysisRunSpec", "AnalysisRunManifest", "execute"),
        "materialize_report": ("ReportSpec", "ReportManifest", "execute"),
        "inspect_evaluation_manifest": (
            "EvaluationRunManifest",
            "EvaluationRunManifest",
            "inspect",
        ),
        "inspect_analysis_manifest": ("AnalysisRunManifest", "AnalysisRunManifest", "inspect"),
        "inspect_report_manifest": ("ReportManifest", "ReportManifest", "inspect"),
        "handoff_evaluation_artifacts": (
            "EvaluationRunManifest",
            "EvaluationRunManifest",
            "handoff",
        ),
        "handoff_analysis_artifacts": ("AnalysisRunManifest", "AnalysisRunManifest", "handoff"),
        "handoff_report_artifacts": ("ReportManifest", "ReportManifest", "handoff"),
    }

    for name, (input_schema, output_schema, action) in expected.items():
        capability = manifest.capabilities[name]
        assert capability.input_schema == input_schema
        assert capability.output_schema == output_schema
        assert capability.action == action

    assert manifest.capabilities["validate_evaluation_spec"].input_schema == "EvaluationRunSpec"
    assert manifest.capabilities["validate_analysis_spec"].input_schema == "AnalysisRunSpec"
    assert manifest.capabilities["validate_report_spec"].input_schema == "ReportSpec"
    assert "trajectory_dataset" in manifest.capabilities["execute_evaluation_run"].artifact_roles
    assert "analysis_table" in manifest.capabilities["execute_analysis_run"].artifact_roles
    assert "report" in manifest.capabilities["materialize_report"].artifact_roles
    assert "report_render" in manifest.capabilities["materialize_report"].artifact_roles
    assert "report_render" in manifest.capabilities["handoff_report_artifacts"].artifact_roles
    assert (
        "artifact_id fields are optional and local URIs remain valid"
        in manifest.capabilities["handoff_report_artifacts"].custody_expectations
    )


def test_provider_manifest_exports_neutral_contract_schema_names() -> None:
    manifest = provider_manifest()
    contract_models = {
        "AnalysisInputRequirement": AnalysisInputRequirement,
        "GraphSpec": GraphSpec,
        "LossTermSpec": LossTermSpec,
        "TaskSpec": TaskSpec,
        "TrainingRunSpec": TrainingRunSpec,
        "TrainingSpec": TrainingSpec,
    }

    for schema_name, model_type in contract_models.items():
        assert manifest.schemas[schema_name] == model_type.model_json_schema()


def test_provider_manifest_graph_spec_schema_exposes_registered_identity() -> None:
    manifest = provider_manifest()
    graph_spec_schema = manifest.schemas["GraphSpec"]
    if graph_spec_schema.get("$ref") == "#/$defs/GraphSpec":
        graph_spec_schema = graph_spec_schema["$defs"]["GraphSpec"]
    properties = graph_spec_schema["properties"]

    assert default_spec_registry.current_version("GraphSpec") == GRAPH_SPEC_SCHEMA_VERSION
    assert properties["schema_id"]["default"] == GRAPH_SPEC_SCHEMA_ID
    assert properties["schema_version"]["default"] == GRAPH_SPEC_SCHEMA_VERSION


def test_provider_validation_exposes_objective_and_studio_migration_entrypoints() -> None:
    objective = validate_spec("objective", {"schema_version": "feedbax.objective.v0"})
    assert not objective.valid
    assert objective.errors[0].type == "invalid_objective_spec"
    assert "feedbax.objective.v0" in objective.errors[0].message

    task_binding = validate_spec(
        "studio_task_binding",
        {
            "schema_version": "feedbax.studio.task_bindings.v1",
            "exposed_outputs": [],
            "bindings": [],
            "metadata": {},
        },
    )
    assert task_binding.valid

    workspace_payload = _schema_workspace().model_dump(mode="json", exclude_none=True)
    workspace_payload["scenarios"]["scenario:train"]["task_binding_spec"] = {
        "schema_version": "feedbax.studio.task_bindings.v0",
        "metadata": {},
    }
    workspace = validate_spec("studio_workspace", workspace_payload)
    assert not workspace.valid
    assert workspace.errors[0].type == "invalid_studio_workspace_spec"
    assert "task_bindings.v0" in workspace.errors[0].message


def test_provider_manifest_exposes_mandible_manifest_mapping_contract() -> None:
    manifest = provider_manifest()
    mappings = manifest.mandible_manifest_mappings

    assert set(mappings) == {
        "GraphSpecManifest",
        "ModelArtifactManifest",
        "TrainingRunSetManifest",
        "TrainingRunManifest",
        "EvaluationRunManifest",
        "CheckpointSelectionManifest",
        "AnalysisRunManifest",
        "ReportManifest",
    }

    training = mappings["TrainingRunManifest"]
    assert training.subject_node_type == "feedbax.training_run"
    assert training.issue_provenance_field == "provenance.issues"
    assert "graph_spec" in training.spec_fields
    assert "training_spec" in training.spec_fields
    assert "artifacts[]" in {field.source_field for field in training.artifact_fields}
    assert "summary_metrics" in training.opaque_domain_fields
    assert "open_in_feedbax_studio" in training.actions
    assert "handoff_artifacts" in training.actions

    model_artifact = mappings["ModelArtifactManifest"]
    store_roles = {
        field.source_field: field.role
        for field in model_artifact.artifact_fields
        if field.source_field.endswith("_store")
    }
    assert model_artifact.subject_node_type == "feedbax.model_artifact"
    assert store_roles == {
        "parameter_store": "model_parameters",
        "state_store": "model_state",
        "optimizer_store": "optimizer_state",
    }
    assert all(
        field.preserves_local_uri and field.optional_artifact_id
        for field in model_artifact.artifact_fields
    )
    assert "parameter_store.roles" in model_artifact.opaque_domain_fields
    assert "mandible/2322726" in model_artifact.related_issue_refs

    checkpoint_selection = mappings["CheckpointSelectionManifest"]
    assert checkpoint_selection.subject_node_type == "feedbax.checkpoint_selection"
    assert checkpoint_selection.spec_fields == ["selection_spec"]
    assert "bank.ref" in checkpoint_selection.parent_ref_fields
    assert "selections[].selected_checkpoint.model_artifact" in (
        checkpoint_selection.parent_ref_fields
    )
    assert "scorer" in checkpoint_selection.opaque_domain_fields


def test_training_run_manifest_mandible_mapping_fixture_preserves_local_refs() -> None:
    run = TrainingRunManifest(
        id="feedbax-training-run:demo",
        status="completed",
        run_set_id="feedbax-training-run-set:demo",
        job_id="demo-job",
        graph_spec=spec_payload("GraphSpec", _minimal_graph_spec(), ref="manifest://graph/demo"),
        training_spec=spec_payload("TrainingSpec", _minimal_training_spec()),
        provenance=Provenance(
            source_repo="https://example.invalid/feedbax-demo.git",
            source_branch="feature/demo",
            source_commit="abc123",
            dirty=False,
            entrypoint=EntrypointRef(kind="feedbax-provider", command="feedbax-provider run-local"),
            issues=["51832b9"],
            parents=[
                ParentRef(
                    kind="GraphSpecManifest",
                    id="feedbax-graph-spec:demo",
                    role="graph_spec",
                    uri="manifests/graph_specs/demo.json",
                )
            ],
        ),
        artifacts=[
            ArtifactRef(
                role="training_history",
                logical_name="history.npz",
                uri="artifacts/demo-job/history.npz",
                storage_backend="feedbax-local",
                metadata={"mandible": {"custody": "handoff-eligible"}},
            )
        ],
        summary_metrics={"final_loss": 0.25},
    )
    mapping = provider_manifest().mandible_manifest_mappings[run.kind]

    assert mapping.subject_node_type == "feedbax.training_run"
    assert mapping.parent_ref_fields == ["graph_spec", "run_set_id", "provenance.parents"]
    assert run.provenance.issues == ["51832b9"]
    assert run.artifacts[0].artifact_id is None
    assert run.artifacts[0].uri == "artifacts/demo-job/history.npz"
    assert run.artifacts[0].metadata["mandible"]["custody"] == "handoff-eligible"


def test_model_artifact_manifest_mandible_mapping_fixture_includes_role_stores() -> None:
    manifest = ModelArtifactManifest(
        id="feedbax-model-artifact:demo",
        status="completed",
        graph_spec=ParentRef(
            kind="GraphSpecManifest",
            id="feedbax-graph-spec:demo",
            role="graph_spec",
            uri="manifests/graph_specs/demo.json",
        ),
        parameter_store=ArrayStoreRef(
            role="params",
            schema_version="feedbax.manifest.array_store.v1",
            storage_backend="npz.v1",
            logical_name="model.arrays.npz",
            artifact_id="mandible-artifact:params-demo",
            uri="artifacts/demo/model.arrays.npz",
            array_count=2,
            roles=[
                "model.network.cell.weight_hh",
                "model.network.readout.bias",
            ],
            metadata={"custody": {"local_uri_authoritative": True}},
        ),
        state_store=ArrayStoreRef(
            role="state",
            schema_version="feedbax.manifest.array_store.v1",
            storage_backend="npz.v1",
            logical_name="model.state.npz",
            uri="artifacts/demo/model.state.npz",
            array_count=1,
            roles=["state.mechanics.position"],
        ),
        optimizer_store=ArrayStoreRef(
            role="optimizer",
            schema_version="feedbax.manifest.array_store.v1",
            storage_backend="npz.v1",
            logical_name="optimizer.npz",
            uri="artifacts/demo/optimizer.npz",
            array_count=1,
            roles=["optimizer.adam.momentum"],
        ),
        provenance=Provenance(
            issues=["51832b9"],
            parents=[ParentRef(kind="TrainingRunManifest", id="feedbax-training-run:demo")],
        ),
    )
    mapping = provider_manifest().mandible_manifest_mappings[manifest.kind]

    mapped_fields = {field.source_field: field for field in mapping.artifact_fields}
    assert mapped_fields["parameter_store"].role == "model_parameters"
    assert mapped_fields["parameter_store"].mandible_artifact_kind == "array_store"
    assert mapped_fields["parameter_store"].optional_artifact_id
    assert manifest.parameter_store is not None
    assert manifest.parameter_store.artifact_id == "mandible-artifact:params-demo"
    assert manifest.parameter_store.uri == "artifacts/demo/model.arrays.npz"
    assert manifest.state_store is not None
    assert manifest.state_store.roles == ["state.mechanics.position"]
    assert manifest.optimizer_store is not None
    assert manifest.optimizer_store.roles == ["optimizer.adam.momentum"]


def test_eval_analysis_report_manifests_preserve_optional_handoff_artifact_ids(
    tmp_path: Path,
) -> None:
    training_parent = ParentRef(
        kind="TrainingRunManifest",
        id="feedbax-training-run:provider-depth",
        role="training_run",
        uri="manifests/training_runs/provider-depth.json",
    )
    evaluation_spec = EvaluationRunSpec(
        evaluation_type="provider_depth_eval",
        inputs=[training_parent],
        params={"split": "validation"},
    )
    evaluation = EvaluationRunManifest(
        id="feedbax-evaluation-run:provider-depth",
        status="completed",
        evaluation_spec=spec_payload(
            "EvaluationRunSpec",
            evaluation_spec.model_dump(mode="json", exclude_none=True),
        ),
        input_training_runs=[training_parent],
        provenance=Provenance(parents=[training_parent], issues=["63c798f"]),
        artifacts=[
            ArtifactRef(
                role="trajectory_dataset",
                logical_name="states.parquet",
                uri="artifacts/eval/states.parquet",
                media_type="application/vnd.apache.parquet",
                metadata={"schema_id": "feedbax.manifest.evaluation_run"},
            )
        ],
    )
    evaluation_path = write_manifest(evaluation, root=tmp_path)

    evaluation_parent = ParentRef(
        kind="EvaluationRunManifest",
        id=evaluation.id,
        role="evaluation_run",
        uri=str(evaluation_path.relative_to(tmp_path)),
    )
    analysis_spec = AnalysisRunSpec(
        analysis_type="feedbax.analysis.plot",
        inputs=[evaluation_parent],
        params={"requested_outputs": ["summary_table"]},
    )
    analysis = AnalysisRunManifest(
        id="feedbax-analysis-run:provider-depth",
        status="completed",
        analysis_spec=spec_payload(
            "AnalysisRunSpec",
            analysis_spec.model_dump(mode="json", exclude_none=True),
        ),
        inputs=[evaluation_parent],
        provenance=Provenance(parents=[evaluation_parent], issues=["63c798f"]),
        artifacts=[
            ArtifactRef(
                role="analysis_table",
                logical_name="summary.csv",
                uri="artifacts/analysis/summary.csv",
                media_type="text/csv",
            )
        ],
    )
    analysis_path = write_manifest(analysis, root=tmp_path)

    analysis_parent = ParentRef(
        kind="AnalysisRunManifest",
        id=analysis.id,
        role="analysis_run",
        uri=str(analysis_path.relative_to(tmp_path)),
    )
    report_spec = ReportSpec(
        report_type="provider_depth_report",
        inputs=[analysis_parent],
        params={"format": "html"},
    )
    report = ReportManifest(
        id="feedbax-report:provider-depth",
        status="completed",
        report_spec=spec_payload(
            "ReportSpec",
            report_spec.model_dump(mode="json", exclude_none=True),
        ),
        inputs=[analysis_parent],
        provenance=Provenance(parents=[analysis_parent], issues=["63c798f"]),
        artifacts=[
            ArtifactRef(
                role="report",
                logical_name="report-bundle.zip",
                uri="artifacts/reports/report-bundle.zip",
                media_type="application/zip",
                metadata={"handoff": {"mandible_artifact_id": None}},
            )
        ],
    )
    report_path = write_manifest(report, root=tmp_path)

    loaded = [load_manifest(path) for path in (evaluation_path, analysis_path, report_path)]
    assert [manifest.kind for manifest in loaded] == [
        "EvaluationRunManifest",
        "AnalysisRunManifest",
        "ReportManifest",
    ]
    assert all(manifest.artifacts[0].artifact_id is None for manifest in loaded)
    assert loaded[0].artifacts[0].uri == "artifacts/eval/states.parquet"
    assert loaded[1].analysis_spec.schema_id == "feedbax.spec.analysis_run"
    assert loaded[2].report_spec.schema_version == "feedbax.spec.report.v1"

    assert validate_spec(
        "evaluation",
        evaluation_spec.model_dump(mode="json", exclude_none=True),
    ).valid
    assert validate_spec("analysis", analysis_spec.model_dump(mode="json", exclude_none=True)).valid
    assert validate_report_spec(report_spec).valid

    index_path = rebuild_manifest_index(tmp_path)
    with sqlite3.connect(index_path) as conn:
        rows = conn.execute(
            """
            SELECT m.kind, a.role, a.artifact_id, a.uri
            FROM manifests AS m
            JOIN artifacts AS a ON a.manifest_id = m.id
            WHERE m.id IN (?, ?, ?)
            ORDER BY m.kind
            """,
            (evaluation.id, analysis.id, report.id),
        ).fetchall()

    assert rows == [
        ("AnalysisRunManifest", "analysis_table", None, "artifacts/analysis/summary.csv"),
        ("EvaluationRunManifest", "trajectory_dataset", None, "artifacts/eval/states.parquet"),
        ("ReportManifest", "report", None, "artifacts/reports/report-bundle.zip"),
    ]


def test_component_registry_snapshot_wraps_existing_registry() -> None:
    snapshot = component_registry_snapshot()
    type_ids = {entry.type_id for entry in snapshot.entries}

    assert snapshot.kind == "components"
    assert "feedbax.component.Gain" in type_ids
    gain = next(entry for entry in snapshot.entries if entry.type_id == "feedbax.component.Gain")
    assert gain.input_ports == ["input"]
    assert gain.output_ports == ["output"]
    assert gain.component_type_id == "Gain"
    assert gain.owner == "feedbax"
    assert gain.package == "feedbax"
    assert gain.provenance == "feedbax"
    assert gain.provenance_kind == "feedbax"
    assert gain.param_schema_version == "1"
    assert gain.supported_param_schema_versions == ["1"]
    assert gain.identity is not None
    assert gain.identity.stable
    assert gain.identity.owner == "feedbax"
    assert gain.migrations == []


def test_validation_functions_accept_small_vertical_slice_specs() -> None:
    graph = _minimal_graph_spec()
    training = _minimal_training_spec()

    assert validate_graph_spec(graph).valid
    assert validate_training_spec(training, graph_spec=graph).valid
    assert validate_task_spec({"type": "SimpleReaches", "params": {}}).valid
    assert validate_evaluation_spec(
        {"evaluation_type": "default", "training_run_ids": ["feedbax-training-run:test"]}
    ).valid
    assert validate_analysis_spec(
        {
            "analysis_type": "feedbax.analysis.plot",
            "inputs": [{"kind": "TrainingRunManifest", "id": "feedbax-training-run:test"}],
            "input_requirements": [
                {
                    "selector": "graph_output:output",
                    "consumer": {
                        "page_id": "page:analysis",
                        "node_id": "analysis-node:plot",
                        "input_port": "series",
                    },
                }
            ],
        }
    ).valid


def test_analysis_validation_rejects_unknown_input_selector_with_graph_context() -> None:
    result = validate_analysis_spec(
        {
            "analysis_type": "feedbax.analysis.plot",
            "inputs": [{"kind": "TrainingRunManifest", "id": "feedbax-training-run:test"}],
            "input_requirements": [
                {
                    "selector": "port:missing.output",
                    "consumer": {"page_id": "page:analysis", "input_port": "series"},
                }
            ],
        },
        graph_spec=_minimal_graph_spec(),
    )

    assert result.valid is False
    assert result.errors[0].type == "analysis_input_graph_mismatch"
    assert result.errors[0].location == {
        "path": "/analysis/input_requirements/0",
        "selector": "port:missing.output",
    }


def test_analysis_validation_does_not_require_explicit_retained_observable() -> None:
    result = validate_analysis_spec(
        {
            "analysis_type": "feedbax.analysis.plot",
            "inputs": [{"kind": "EvaluationRunManifest", "id": "feedbax-eval-run:test"}],
            "input_requirements": [
                {
                    "selector": "graph_output:output",
                    "retention": {"mode": "trajectory"},
                }
            ],
        },
        graph_spec=_minimal_graph_spec(),
    )

    assert result.valid is True


def test_task_validation_rejects_dense_delayed_reach_trajectory_params() -> None:
    result = validate_task_spec(
        {
            "type": "DelayedReaches",
            "params": {
                "n_steps": 140,
                "targets": [[[0.0, 0.0], [0.1, 0.1]]],
                "epoch_len_ranges": [[0, 1], [10, 30]],
                "target_on_epochs": [1, 2],
                "hold_epochs": [0, 1],
                "move_epochs": [2],
            },
        }
    )

    assert not result.valid
    assert result.errors[0].type == "dense_task_trajectory_not_allowed"
    assert result.errors[0].location == {"path": "/params/targets"}


def test_task_validation_rejects_invalid_delayed_reach_epoch_params() -> None:
    result = validate_task_spec(
        {
            "type": "DelayedReaches",
            "params": {
                "epoch_len_ranges": [[10, 1]],
                "target_on_epochs": [2],
                "hold_epochs": [0],
                "move_epochs": [1],
            },
        }
    )

    assert not result.valid
    assert {error.type for error in result.errors} == {
        "invalid_epoch_len_range",
        "invalid_epoch_index",
    }


def test_task_validation_reports_pathful_step_count_errors() -> None:
    invalid = validate_task_spec({"type": "DelayedReaches", "params": {"n_steps": 0}})
    mismatch = validate_task_spec(
        {
            "type": "DelayedReaches",
            "timeline": {"n_steps": 140},
            "params": {"n_steps": 120},
        }
    )

    assert invalid.errors[0].type == "invalid_task_n_steps"
    assert invalid.errors[0].location == {"path": "/params/n_steps"}
    assert {error.type for error in mismatch.errors} == {"task_n_steps_mismatch"}


def test_delayed_center_out_preset_task_spec_validates_and_infers_control_stages() -> None:
    payload = {
        "type": "DelayedReaches",
        "params": {
            "preset": "delayed_center_out",
            "n_control_stages": 140,
            "workspace": [[-1.0, -1.0], [1.0, 1.0]],
            "epoch_len_ranges": [[20, 60]],
            "p_catch_trial": 0.25,
        },
    }

    result = validate_task_spec(payload)

    assert result.valid is True
    assert infer_task_n_steps(payload) == 140


def test_delayed_center_out_validation_rejects_inconsistent_control_stages() -> None:
    result = validate_task_spec(
        {
            "type": "DelayedReaches",
            "params": {
                "preset": "delayed_center_out",
                "n_control_stages": 140,
                "n_steps": 140,
            },
        }
    )

    assert result.valid is False
    assert {error.type for error in result.errors} == {"task_n_steps_mismatch"}


def test_delayed_reaches_validation_rejects_invalid_metadata_policy() -> None:
    result = validate_task_spec(
        {
            "type": "DelayedReaches",
            "params": {
                "epoch_names": ["prep", "movement"],
                "epoch_len_ranges": [[20, 60]],
                "target_on_epochs": [0, 1],
                "catch_metadata_policy": "trial_type",
            },
        }
    )

    assert result.valid is False
    assert {error.type for error in result.errors} == {"invalid_catch_metadata_policy"}


def test_graph_validation_reports_unknown_components() -> None:
    graph = _minimal_graph_spec()
    graph["nodes"]["bad"] = {
        "type": "MissingComponent",
        "params": {},
        "input_ports": [],
        "output_ports": [],
    }

    result = validate_graph_spec(graph)

    assert not result.valid
    assert result.errors[0].type == "unknown_component_type"


def test_graph_validation_rejects_runtime_network_authoring_payloads() -> None:
    result = validate_graph_spec(_runtime_network_graph_spec())

    assert not result.valid
    assert "unknown_component_type" in {error.type for error in result.errors}


def test_graph_validation_rejects_task_nodes() -> None:
    graph = _minimal_graph_spec()
    graph["nodes"]["task"] = {
        "type": "SimpleReaches",
        "params": {},
        "input_ports": [],
        "output_ports": ["inputs", "targets", "inits", "intervene"],
    }

    result = validate_graph_spec(graph)

    assert not result.valid
    assert result.errors[0].type == "task_node_not_allowed"


def test_graph_validation_rejects_degenerate_single_input_mux() -> None:
    graph = GraphSpec(
        nodes={
            "source": {
                "type": "Constant",
                "params": {"value": [1.0]},
                "input_ports": [],
                "output_ports": ["output"],
            },
            "mux": {
                "type": "Mux",
                "params": {"n_inputs": 2},
                "input_ports": ["in_0", "in_1"],
                "output_ports": ["output"],
            },
        },
        wires=[
            {
                "source_node": "source",
                "source_port": "output",
                "target_node": "mux",
                "target_port": "in_0",
            }
        ],
        output_ports=["output"],
        output_bindings={"output": ("mux", "output")},
    )

    result = validate_graph_spec(graph)

    assert not result.valid
    mux_error = next(
        error for error in result.errors if error.type == "mux_needs_two_connected_inputs"
    )
    assert mux_error.message == "Mux 'mux' needs at least two connected inputs"


def test_graph_validation_uses_schema_for_direction_occupied_and_dtype_mismatch() -> None:
    graph = {
        "nodes": {
            "linear": {
                "type": "Linear",
                "params": {},
                "input_ports": ["input"],
                "output_ports": ["output"],
            },
            "muscle": {
                "type": "ReluMuscle",
                "params": {},
                "input_ports": ["excitation"],
                "output_ports": ["force", "activation"],
            },
        },
        "wires": [
            {
                "source_node": "linear",
                "source_port": "output",
                "target_node": "muscle",
                "target_port": "excitation",
            },
            {
                "source_node": "linear",
                "source_port": "input",
                "target_node": "muscle",
                "target_port": "excitation",
            },
        ],
        "input_ports": [],
        "output_ports": [],
        "input_bindings": {},
        "output_bindings": {},
    }

    result = validate_graph_spec(graph)
    issue_types = {issue.type for issue in result.errors}

    assert not result.valid
    assert "graph_wire_dtype_mismatch" in issue_types
    assert "wrong_source_port_direction" in issue_types
    assert "graph_input_occupied" in issue_types


def test_graph_validation_reports_network_missing_subgraph_before_build() -> None:
    graph = GraphSpec(
        nodes={
            "network": ComponentSpec(
                type="Network",
                params={"input_size": 1, "hidden_size": 4, "out_size": 1},
                input_ports=["input"],
                output_ports=["output"],
            )
        },
        input_ports=["input"],
        output_ports=["output"],
        input_bindings={"input": ("network", "input")},
        output_bindings={"output": ("network", "output")},
    )

    result = validate_graph_spec(graph)
    issue_types = {issue.type for issue in result.errors}

    assert not result.valid
    assert "missing_subgraph" in issue_types


def test_studio_task_timeline_spec_validates_value_specs() -> None:
    timeline = StudioTaskTimelineSpec.model_validate(
        {
            "schema_version": "feedbax.spec.studio.task_timeline.v1",
            "epochs": [
                {
                    "id": "epoch:0",
                    "label": "hold",
                    "index": 0,
                    "length": {
                        "schema_version": "feedbax.spec.studio.value.v1",
                        "mode": "constant",
                        "value": {"min": 0, "max": 1},
                        "metadata": {"scope": "trial"},
                    },
                    "metadata": {},
                }
            ],
            "signals": [
                {
                    "id": "hold",
                    "label": "Hold cue",
                    "kind": "signal",
                    "task_data_id": "inputs",
                    "path": "inputs.hold",
                    "epoch_ids": ["epoch:0"],
                    "value_spec": {
                        "schema_version": "feedbax.spec.studio.value.v1",
                        "mode": "distribution",
                        "distribution": {
                            "family": "uniform",
                            "parameters": {"min": 0.0, "max": 1.0},
                        },
                        "sampling_scope": "trial",
                        "metadata": {},
                    },
                    "value_schema": {
                        "id": "schema:inputs",
                        "label": "Inputs",
                        "kind": "array",
                        "dtype": "float",
                        "shape": [2],
                        "origin": "declared",
                        "metadata": {},
                    },
                    "task_data_schema": {
                        "id": "task_data:inputs",
                        "label": "Inputs",
                        "kind": "signal",
                        "role": "model_input",
                        "path": "inputs",
                        "bindable": True,
                        "value_schema": {
                            "id": "schema:inputs",
                            "label": "Inputs",
                            "kind": "array",
                            "origin": "declared",
                            "metadata": {},
                        },
                        "origin": "declared",
                        "metadata": {},
                    },
                    "metadata": {"value_spec_modes": ["constant", "distribution"]},
                }
            ],
            "segments": [
                {
                    "id": "cue_window",
                    "label": "cue window",
                    "epoch_ids": ["epoch:0"],
                    "metadata": {},
                }
            ],
            "metadata": {"task_type": "DelayedReaches"},
        }
    )

    assert timeline.epochs[0].length.mode == "constant"
    assert timeline.signals[0].epoch_ids == ["epoch:0"]
    assert timeline.signals[0].value_spec is not None
    assert timeline.signals[0].value_spec.mode == "distribution"
    assert timeline.signals[0].value_schema["shape"] == [2]
    assert timeline.signals[0].task_data_schema["role"] == "model_input"
    assert timeline.segments[0].id == "cue_window"


def test_training_manifest_writes_artifacts_and_rebuildable_index(tmp_path: Path) -> None:
    checkpoint = tmp_path / "model.eqx"
    checkpoint.write_bytes(b"checkpoint bytes")

    root = tmp_path / "runs"
    manifest, path = write_training_run_manifest(
        job_id="job-1",
        total_batches=2,
        training_spec=_minimal_training_spec(),
        task_spec={"type": "SimpleReaches", "params": {}},
        task_binding_spec={
            "schema_version": "feedbax.spec.studio.task_bindings.v2",
            "exposed_data": [],
            "bindings": [],
            "metadata": {},
        },
        graph_spec=_minimal_graph_spec(),
        checkpoint_path=checkpoint,
        history_events=[{"type": "training_progress", "batch": 1, "loss": 0.5}],
        final_loss=0.25,
        root=root,
        provenance=Provenance(source_commit="abc123", dirty=False),
        issues=["5429a23"],
    )

    assert path.exists()
    loaded = load_manifest(path)
    assert loaded.id == manifest.id
    assert loaded.status == "completed"
    assert loaded.summary_metrics["final_loss"] == 0.25
    assert loaded.task_binding_spec is not None
    assert loaded.provenance.issues == ["5429a23"]

    checkpoint_ref = next(
        artifact for artifact in loaded.artifacts if artifact.role == "training_checkpoint"
    )
    assert checkpoint_ref.sha256 == sha256_file(checkpoint)
    assert checkpoint_ref.uri is not None
    assert Path(checkpoint_ref.uri).exists()

    history_ref = next(
        artifact for artifact in loaded.artifacts if artifact.role == "training_history"
    )
    assert history_ref.media_type == "application/json"
    assert history_ref.uri is not None
    assert Path(history_ref.uri).exists()

    db_path = rebuild_manifest_index(root)
    with sqlite3.connect(db_path) as conn:
        manifest_count = conn.execute("SELECT COUNT(*) FROM manifests").fetchone()[0]
        artifact_count = conn.execute("SELECT COUNT(*) FROM artifacts").fetchone()[0]

    assert manifest_count == 1
    assert artifact_count == 2


def test_provider_http_endpoints() -> None:
    client = TestClient(create_app())

    health = client.get("/api/provider/health")
    assert health.status_code == 200
    assert health.json()["provider"] == "feedbax"

    validation = client.post(
        "/api/provider/validate/graph",
        json={"spec": _minimal_graph_spec()},
    )
    assert validation.status_code == 200
    assert validation.json()["valid"] is True

    schemas = client.post(
        "/api/provider/studio/schemas",
        json={
            "workspace": _schema_workspace().model_dump(mode="json", exclude_none=True),
            "scenario_id": "scenario:train",
        },
    )
    assert schemas.status_code == 200
    payload = schemas.json()
    assert payload["kind"] == "studio_schema_registry"
    assert payload["scenario_id"] == "scenario:train"
    assert any(port["id"] == "port:network.input:input" for port in payload["ports"])
    assert any(item["id"] == "task_data:inputs" for item in payload["task_data"])
    assert any(
        target["selector"] == "path:states.mechanics.effector.pos"
        for target in payload["selector_targets"]
    )
    assert any(issue["type"] == "stage_missing_scenario" for issue in payload["issues"])

    runtime_schemas = client.post(
        "/api/provider/studio/schemas",
        json={
            "workspace": _schema_workspace().model_dump(mode="json", exclude_none=True),
            "scenario_id": "scenario:train",
            "runtime_introspection": {"enabled": True, "max_targets": 4},
        },
    )
    assert runtime_schemas.status_code == 200
    runtime_payload = runtime_schemas.json()
    assert runtime_payload["metadata"]["runtime_introspection"]["status"] == "unavailable"
    assert any(
        issue["type"] == "runtime_introspection_unavailable" and issue["severity"] == "warning"
        for issue in runtime_payload["issues"]
    )


def test_studio_schema_enumeration_returns_ports_task_data_targets_and_issues() -> None:
    registry = enumerate_studio_schema_registry(_schema_workspace(), "scenario:train")

    assert registry.workspace_id is not None
    assert registry.scenario_id == "scenario:train"
    assert any(port.id == "port:network.input:input" for port in registry.ports)
    bound = next(port for port in registry.ports if port.id == "port:network.input:input")
    assert bound.bound_task_data_id == "task_data:inputs"
    assert any(item.id == "task_data:targets" for item in registry.task_data)
    selectors = {target.selector for target in registry.selector_targets}
    assert "port:network.output" in selectors
    assert "task_data:inputs" in selectors
    assert "probe:activation-tap" in selectors
    assert "probe:manual-probe" in selectors
    assert "path:states.mechanics.effector.pos" in selectors
    assert any(issue.type == "stage_missing_scenario" for issue in registry.issues)
    assert registry.metadata["runtime_introspection"]["status"] == "not_requested"


def test_studio_schema_enumeration_reports_workspace_migration_rejection() -> None:
    workspace = _schema_workspace().model_dump(mode="json", exclude_none=True)
    workspace["scenarios"]["scenario:train"]["task_binding_spec"] = {
        "schema_version": "feedbax.studio.task_bindings.v0",
        "metadata": {},
    }

    registry = enumerate_studio_schema_registry(workspace, "scenario:train")

    assert registry.ports == []
    assert registry.issues[0].type == "workspace_schema_version_error"
    assert "task_bindings.v0" in registry.issues[0].message


def test_studio_schema_enumeration_does_not_wrap_runtime_network_ports() -> None:
    workspace = build_default_studio_workspace(
        label="Runtime network",
        graph=GraphSpec.model_validate(_runtime_network_graph_spec()),
    )
    train_stage = next(stage for stage in workspace.stages if stage.kind == "train")
    scenario = workspace.scenarios[train_stage.scenario_id]
    scenario.task_binding_spec = StudioTaskBindingSpec.model_validate(
        {
            "schema_version": "feedbax.spec.studio.task_bindings.v2",
            "exposed_data": [
                {
                    "id": "inputs",
                    "label": "Inputs",
                    "kind": "signal",
                    "path": "inputs",
                    "bindable": True,
                    "metadata": {},
                }
            ],
            "bindings": [
                {
                    "id": "task:inputs->network:target",
                    "source_data_id": "inputs",
                    "target_node_id": "network",
                    "target_port": "target",
                    "role": "model_input",
                    "metadata": {},
                }
            ],
            "metadata": {},
        }
    )

    registry = enumerate_studio_schema_registry(workspace, train_stage.scenario_id)

    assert not any(port.id == "port:network.input:input" for port in registry.ports)
    assert any(issue.type == "task_binding_unknown_schema" for issue in registry.issues)


def test_runtime_wrapper_normalization_does_not_generate_recurrent_cell_edges() -> None:
    graph = GraphSpec.model_validate(_runtime_network_graph_spec())
    normalized = normalize_graph_for_studio_authoring(graph)

    assert normalized.nodes["network"].type == "SimpleStagedNetwork"
    assert normalized.subgraphs is None


def test_runtime_wrapper_normalization_does_not_lower_hidden_population_constraints() -> None:
    raw_graph = _runtime_network_graph_spec()
    raw_graph["nodes"]["network"]["params"]["hidden_size"] = 4
    raw_graph["nodes"]["network"]["params"]["input_size"] = 2
    raw_graph["nodes"]["network"]["params"]["population_structure"] = {
        "schema_id": "feedbax.spec.population_structure",
        "schema_version": "feedbax.spec.population_structure.v1",
        "assignment": "explicit",
        "n_input_only": 1,
        "n_readout_only": 1,
        "n_recurrent_only": 1,
        "n_input_readout": 1,
        "input_only_indices": [0],
        "readout_only_indices": [1],
        "recurrent_only_indices": [2],
        "input_readout_indices": [3],
    }
    graph = GraphSpec.model_validate(raw_graph)

    normalized = normalize_graph_for_studio_authoring(graph)

    assert normalized.subgraphs is None
    assert normalized.parameter_constraints == []


def test_runtime_wrapper_normalization_preserves_legacy_model_wrapper() -> None:
    legacy_inner = GraphSpec(
        nodes={
            "cell": {
                "type": "GRU",
                "params": {"input_size": 4, "hidden_size": 100},
                "input_ports": ["input", "hidden"],
                "output_ports": ["output", "hidden"],
            }
        },
        wires=[],
        input_ports=["input"],
        output_ports=["output"],
        input_bindings={"input": ("cell", "input")},
        output_bindings={"output": ("cell", "output")},
    )
    wrapped = GraphSpec(
        nodes={
            "network": {
                "type": "Network",
                "params": {},
                "input_ports": ["input", "feedback"],
                "output_ports": ["output"],
            }
        },
        subgraphs={
            "network": GraphSpec(
                nodes={
                    "model": {
                        "type": "Subgraph",
                        "params": {},
                        "input_ports": ["input", "feedback"],
                        "output_ports": ["output"],
                    }
                },
                wires=[],
                input_ports=["input", "feedback"],
                output_ports=["output"],
                input_bindings={
                    "input": ("model", "input"),
                    "feedback": ("model", "feedback"),
                },
                output_bindings={"output": ("model", "output")},
                subgraphs={"model": legacy_inner},
            ),
        },
    )

    subgraph = normalize_graph_for_studio_authoring(wrapped).subgraphs["network"]

    assert subgraph.nodes["model"].type == "Subgraph"
    assert subgraph.subgraphs["model"].nodes["cell"].type == "GRU"
    assert subgraph.output_bindings == {"output": ("model", "output")}


def test_runtime_wrapper_normalization_does_not_mark_feedback_cut_recurrent() -> None:
    graph = GraphSpec(
        nodes={
            "network": {
                "type": "Network",
                "params": {},
                "input_ports": ["input", "feedback"],
                "output_ports": ["output"],
            },
            "mechanics": {
                "type": "PointMass",
                "params": {},
                "input_ports": ["force"],
                "output_ports": ["effector"],
            },
            "feedback": {
                "type": "FeedbackChannels",
                "params": {},
                "input_ports": ["input"],
                "output_ports": ["output"],
            },
        },
        wires=[
            {
                "source_node": "network",
                "source_port": "output",
                "target_node": "mechanics",
                "target_port": "force",
            },
            {
                "source_node": "mechanics",
                "source_port": "effector",
                "target_node": "feedback",
                "target_port": "input",
            },
            {
                "source_node": "feedback",
                "source_port": "output",
                "target_node": "network",
                "target_port": "feedback",
            },
        ],
    )

    recurrent_wire = normalize_graph_for_studio_authoring(graph).wires[2]

    assert recurrent_wire.temporality == "instant"
    assert recurrent_wire.recurrent_initializer is None


def test_graph_connection_schema_rejects_instant_cycles_and_accepts_recurrent_cut() -> None:
    graph = GraphSpec(
        nodes={
            "a": {
                "type": "Gain",
                "params": {},
                "input_ports": ["input"],
                "output_ports": ["output"],
            },
            "b": {
                "type": "Gain",
                "params": {},
                "input_ports": ["input"],
                "output_ports": ["output"],
            },
        },
        wires=[
            {
                "source_node": "a",
                "source_port": "output",
                "target_node": "b",
                "target_port": "input",
            },
            {
                "source_node": "b",
                "source_port": "output",
                "target_node": "a",
                "target_port": "input",
            },
        ],
    )

    issues = validate_graph_connection_schema(graph)
    assert "instant_cycle" in {issue.type for issue in issues}

    recurrent_graph = graph.model_copy(
        update={
            "wires": [
                graph.wires[0],
                graph.wires[1].model_copy(
                    update={
                        "temporality": "recurrent",
                        "recurrent_initializer": {
                            "kind": "zeros",
                            "scope": "trial",
                            "shape": [1],
                        },
                    }
                ),
            ]
        }
    )
    recurrent_issues = validate_graph_connection_schema(recurrent_graph)
    assert "instant_cycle" not in {issue.type for issue in recurrent_issues}
    assert "recurrent_initializer_missing" not in {issue.type for issue in recurrent_issues}


def test_studio_schema_enumeration_reports_dynamic_mux_input_mismatch() -> None:
    graph = GraphSpec(
        nodes={
            "mux": {
                "type": "Mux",
                "params": {"n_inputs": 2},
                "input_ports": ["in_0", "in_1"],
                "output_ports": ["output"],
            }
        },
        output_ports=["output"],
        output_bindings={"output": ("mux", "output")},
    )
    workspace = build_default_studio_workspace(label="Mux schema", graph=graph)
    train_stage = next(stage for stage in workspace.stages if stage.kind == "train")
    scenario = workspace.scenarios[train_stage.scenario_id]
    scenario.task_binding_spec = StudioTaskBindingSpec.model_validate(
        {
            "schema_version": "feedbax.spec.studio.task_bindings.v2",
            "exposed_data": [
                {
                    "id": "target_on",
                    "label": "Target shown",
                    "kind": "signal",
                    "role": "model_input",
                    "path": "inputs.target_on",
                    "bindable": True,
                    "dtype": "float32",
                    "expected_shape": ["time", 1],
                    "metadata": {},
                }
            ],
            "bindings": [
                {
                    "id": "task:target_on->mux:in_2",
                    "source_data_id": "target_on",
                    "target_node_id": "mux",
                    "target_port": "in_2",
                    "role": "model_input",
                    "metadata": {},
                }
            ],
            "metadata": {},
        }
    )

    registry = enumerate_studio_schema_registry(workspace, train_stage.scenario_id)
    issue_types = {issue.type for issue in registry.issues}

    assert "dynamic_port_arity_mismatch" in issue_types
    assert "unknown_task_binding_target_port" in issue_types
    assert not any(port.id == "port:mux.in_2:input" for port in registry.ports)


def test_studio_schema_enumeration_reports_dynamic_demux_output_mismatch() -> None:
    graph = GraphSpec(
        nodes={
            "split": {
                "type": "Demux",
                "params": {"sizes": [2, 1, 3]},
                "input_ports": ["input"],
                "output_ports": ["out_0", "out_1"],
            }
        },
        input_ports=["input"],
        output_ports=["tail"],
        input_bindings={"input": ("split", "input")},
        output_bindings={"tail": ("split", "out_2")},
    )
    workspace = build_default_studio_workspace(label="Demux schema", graph=graph)
    train_stage = next(stage for stage in workspace.stages if stage.kind == "train")

    registry = enumerate_studio_schema_registry(workspace, train_stage.scenario_id)
    issue = next(issue for issue in registry.issues if issue.type == "dynamic_port_arity_mismatch")

    assert "Demux node 'split'" in issue.message
    assert not any(port.id == "port:split.out_2:output" for port in registry.ports)


def test_studio_schema_task_data_trajectory_bindings_use_sample_view() -> None:
    graph = GraphSpec(
        nodes={
            "network": {
                "type": "Network",
                "params": {"input_size": 2, "hidden_size": 100, "out_size": 6},
                "input_ports": ["input"],
                "output_ports": ["output"],
            }
        }
    )
    workspace = build_default_studio_workspace(label="Sample view schema", graph=graph)
    train_stage = next(stage for stage in workspace.stages if stage.kind == "train")
    scenario = workspace.scenarios[train_stage.scenario_id]
    scenario.task_binding_spec = StudioTaskBindingSpec.model_validate(
        {
            "schema_version": "feedbax.spec.studio.task_bindings.v2",
            "exposed_data": [
                {
                    "id": "target_position",
                    "label": "Target position",
                    "kind": "signal",
                    "role": "model_input",
                    "path": "inputs.effector_target.pos",
                    "bindable": True,
                    "dtype": "float32",
                    "expected_shape": ["time", 2],
                    "metadata": {},
                }
            ],
            "bindings": [
                {
                    "id": "task:target_position->network:input",
                    "source_data_id": "target_position",
                    "target_node_id": "network",
                    "target_port": "input",
                    "role": "model_input",
                    "metadata": {},
                }
            ],
            "metadata": {},
        }
    )

    registry = enumerate_studio_schema_registry(workspace, train_stage.scenario_id)
    task_data = next(item for item in registry.task_data if item.id == "task_data:target_position")
    network_input = next(port for port in registry.ports if port.id == "port:network.input:input")

    assert task_data.value_schema.shape == ["time", 2]
    assert task_data.value_schema.metadata["sample_shape"] == [2]
    assert task_data.value_schema.metadata["time_axis"] == 0
    assert network_input.value_schema.shape is None
    issue_types = {issue.type for issue in registry.issues}
    assert "task_binding_rank_mismatch" not in issue_types
    assert "task_binding_shape_mismatch" not in issue_types
    assert "task_binding_dtype_mismatch" not in issue_types


def test_studio_schema_enumeration_infers_mux_output_width_from_sample_shapes() -> None:
    graph = GraphSpec(
        nodes={
            "mux": {
                "type": "Mux",
                "params": {"n_inputs": 2},
                "input_ports": ["in_0", "in_1"],
                "output_ports": ["output"],
            }
        },
        output_ports=["output"],
        output_bindings={"output": ("mux", "output")},
    )
    workspace = build_default_studio_workspace(label="Mux width schema", graph=graph)
    train_stage = next(stage for stage in workspace.stages if stage.kind == "train")
    scenario = workspace.scenarios[train_stage.scenario_id]
    scenario.task_binding_spec = StudioTaskBindingSpec.model_validate(
        {
            "schema_version": "feedbax.spec.studio.task_bindings.v2",
            "exposed_data": [
                {
                    "id": "target_position",
                    "label": "Target position",
                    "kind": "signal",
                    "role": "model_input",
                    "path": "inputs.effector_target.pos",
                    "bindable": True,
                    "dtype": "float32",
                    "expected_shape": ["time", 2],
                    "metadata": {},
                },
                {
                    "id": "hold",
                    "label": "Hold/go cue",
                    "kind": "signal",
                    "role": "model_input",
                    "path": "inputs.hold",
                    "bindable": True,
                    "dtype": "float32",
                    "expected_shape": ["time", 1],
                    "metadata": {},
                },
            ],
            "bindings": [
                {
                    "id": "task:target_position->mux:in_0",
                    "source_data_id": "target_position",
                    "target_node_id": "mux",
                    "target_port": "in_0",
                    "role": "model_input",
                    "metadata": {},
                },
                {
                    "id": "task:hold->mux:in_1",
                    "source_data_id": "hold",
                    "target_node_id": "mux",
                    "target_port": "in_1",
                    "role": "model_input",
                    "metadata": {},
                },
            ],
            "metadata": {},
        }
    )

    registry = enumerate_studio_schema_registry(workspace, train_stage.scenario_id)
    mux_output = next(port for port in registry.ports if port.id == "port:mux.output:output")

    assert mux_output.value_schema.shape == [3]
    assert mux_output.value_schema.rank == 1
    assert mux_output.value_schema.metadata["dimension_source"] == "mux_concat_inputs"
    assert "mux_needs_two_connected_inputs" not in {issue.type for issue in registry.issues}


def test_studio_schema_reports_derived_dimension_conflict() -> None:
    graph = GraphSpec(
        nodes={
            "mux": {
                "type": "Mux",
                "params": {"n_inputs": 2},
                "input_ports": ["in_0", "in_1"],
                "output_ports": ["output"],
            },
            "cell": {
                "type": "GRU",
                "params": {"input_size": 7, "hidden_size": 5},
                "input_ports": ["input", "hidden"],
                "output_ports": ["output", "hidden"],
            },
        },
        wires=[
            {
                "source_node": "mux",
                "source_port": "output",
                "target_node": "cell",
                "target_port": "input",
            }
        ],
        derived_dimensions=[
            {
                "node": "cell",
                "param": "input_size",
                "port": "input",
                "metadata": {"dimension_source": "mux_concat_inputs"},
            }
        ],
    )
    workspace = build_default_studio_workspace(label="Derived dimension conflict", graph=graph)
    train_stage = next(stage for stage in workspace.stages if stage.kind == "train")
    scenario = workspace.scenarios[train_stage.scenario_id]
    scenario.task_binding_spec = StudioTaskBindingSpec.model_validate(
        {
            "schema_version": "feedbax.spec.studio.task_bindings.v2",
            "exposed_data": [
                {
                    "id": "position",
                    "label": "Position",
                    "kind": "signal",
                    "role": "model_input",
                    "path": "inputs.position",
                    "bindable": True,
                    "dtype": "float32",
                    "expected_shape": ["time", 2],
                    "metadata": {},
                },
                {
                    "id": "cue",
                    "label": "Cue",
                    "kind": "signal",
                    "role": "model_input",
                    "path": "inputs.cue",
                    "bindable": True,
                    "dtype": "float32",
                    "expected_shape": ["time", 1],
                    "metadata": {},
                },
            ],
            "bindings": [
                {
                    "id": "task:position->mux:in_0",
                    "source_data_id": "position",
                    "target_node_id": "mux",
                    "target_port": "in_0",
                    "role": "model_input",
                    "metadata": {},
                },
                {
                    "id": "task:cue->mux:in_1",
                    "source_data_id": "cue",
                    "target_node_id": "mux",
                    "target_port": "in_1",
                    "role": "model_input",
                    "metadata": {},
                },
            ],
            "metadata": {},
        }
    )

    registry = enumerate_studio_schema_registry(workspace, train_stage.scenario_id)
    conflict = next(issue for issue in registry.issues if issue.type == "derived_dimension_conflict")

    assert "declared 7" in conflict.message
    assert "derived 3" in conflict.message
    assert conflict.location["path"].endswith("/graph/derived_dimensions/0")


def test_studio_schema_uses_subgraph_boundary_shapes_for_parent_ports() -> None:
    child_graph = GraphSpec(
        nodes={
            "cell": {
                "type": "GRU",
                "params": {"input_size": 4, "hidden_size": 100},
                "input_ports": ["input", "hidden"],
                "output_ports": ["output", "hidden"],
            },
            "readout": {
                "type": "Linear",
                "params": {"input_size": 100, "output_size": 2},
                "input_ports": ["input"],
                "output_ports": ["output"],
            },
        },
        wires=[
            {
                "source_node": "cell",
                "source_port": "output",
                "target_node": "readout",
                "target_port": "input",
            }
        ],
        input_ports=["input", "hidden"],
        output_ports=["output", "hidden"],
        input_bindings={"input": ("cell", "input"), "hidden": ("cell", "hidden")},
        output_bindings={"output": ("readout", "output"), "hidden": ("cell", "output")},
    )
    graph = GraphSpec(
        nodes={
            "task_mux": {
                "type": "Mux",
                "params": {"n_inputs": 3},
                "input_ports": ["in_0", "in_1", "in_2"],
                "output_ports": ["output"],
            },
            "network": {
                "type": "Subgraph",
                "params": {},
                "input_ports": ["input", "hidden"],
                "output_ports": ["output", "hidden"],
            },
        },
        subgraphs={"network": child_graph},
    )
    workspace = build_default_studio_workspace(label="Subgraph boundary schema", graph=graph)
    train_stage = next(stage for stage in workspace.stages if stage.kind == "train")
    scenario = workspace.scenarios[train_stage.scenario_id]
    scenario.task_binding_spec = StudioTaskBindingSpec.model_validate(
        {
            "schema_version": "feedbax.spec.studio.task_bindings.v2",
            "exposed_data": [
                {
                    "id": "target_position",
                    "label": "Target position",
                    "kind": "signal",
                    "role": "model_input",
                    "path": "inputs.effector_target",
                    "bindable": True,
                    "dtype": "float32",
                    "expected_shape": ["time", 4],
                    "metadata": {},
                },
                {
                    "id": "hold",
                    "label": "Hold/go cue",
                    "kind": "signal",
                    "role": "model_input",
                    "path": "inputs.hold",
                    "bindable": True,
                    "dtype": "float32",
                    "expected_shape": ["time", 1],
                    "metadata": {},
                },
                {
                    "id": "target_on",
                    "label": "Target shown",
                    "kind": "signal",
                    "role": "model_input",
                    "path": "inputs.target_on",
                    "bindable": True,
                    "dtype": "float32",
                    "expected_shape": ["time", 1],
                    "metadata": {},
                },
            ],
            "bindings": [
                {
                    "id": "task:target_position->task_mux:in_0",
                    "source_data_id": "target_position",
                    "target_node_id": "task_mux",
                    "target_port": "in_0",
                    "role": "model_input",
                    "metadata": {},
                },
                {
                    "id": "task:hold->task_mux:in_1",
                    "source_data_id": "hold",
                    "target_node_id": "task_mux",
                    "target_port": "in_1",
                    "role": "model_input",
                    "metadata": {},
                },
                {
                    "id": "task:target_on->task_mux:in_2",
                    "source_data_id": "target_on",
                    "target_node_id": "task_mux",
                    "target_port": "in_2",
                    "role": "model_input",
                    "metadata": {},
                },
            ],
            "metadata": {},
        }
    )

    registry = enumerate_studio_schema_registry(workspace, train_stage.scenario_id)
    mux_output = next(port for port in registry.ports if port.id == "port:task_mux.output:output")
    feedback_input = next(port for port in registry.ports if port.id == "port:network.input:input")
    hidden_inputs = [port for port in registry.ports if port.id == "port:network.hidden:input"]

    assert mux_output.value_schema.shape == [6]
    assert feedback_input.value_schema.dtype == "vector"
    assert len(hidden_inputs) == 1
    assert hidden_inputs[0].value_schema.dtype == "vector"


def test_studio_schema_enumeration_runtime_introspection_hook_adds_sample_leaf_targets() -> None:
    def introspector(workspace, scenario_id, options):
        assert workspace.id
        assert scenario_id == "scenario:train"
        assert options.max_targets == 1
        return RuntimeIntrospectionResult(
            sample_leaves=[
                RuntimeSampleLeafSchema(
                    path="states.network.hidden",
                    label="Network hidden state",
                    value=[[0.0, 1.0], [2.0, 3.0]],
                    source={"sample": "representative"},
                ),
                RuntimeSampleLeafSchema(
                    path="task.validation_trials.targets",
                    label="Validation targets",
                    dtype="float32",
                    shape=[8, 2],
                ),
            ],
            metadata={"provider_hook": "test"},
        )

    registry = enumerate_studio_schema_registry(
        _schema_workspace(),
        "scenario:train",
        runtime_introspection={"enabled": True, "max_targets": 1},
        runtime_introspector=introspector,
    )

    runtime_targets = [
        target for target in registry.selector_targets if target.origin == "runtime_sample"
    ]
    assert len(runtime_targets) == 1
    target = runtime_targets[0]
    assert target.kind == "sample_leaf"
    assert target.selector == "path:states.network.hidden"
    assert target.value_schema.shape == [2, 2]
    assert target.value_schema.rank == 2
    assert target.value_schema.origin == "runtime_sample"
    assert registry.metadata["runtime_introspection"]["status"] == "completed"
    assert registry.metadata["runtime_introspection"]["target_count"] == 1
    assert registry.metadata["runtime_introspection"]["truncated"] is True


def test_studio_schema_enumeration_runtime_introspection_failure_is_warning() -> None:
    def introspector(_workspace, _scenario_id, _options):
        raise RuntimeError("sample unavailable")

    registry = enumerate_studio_schema_registry(
        _schema_workspace(),
        "scenario:train",
        runtime_introspection=True,
        runtime_introspector=introspector,
    )

    issue = next(issue for issue in registry.issues if issue.type == "runtime_introspection_failed")
    assert issue.severity == "warning"
    assert "sample unavailable" in issue.message
    assert registry.metadata["runtime_introspection"]["status"] == "failed"


def test_studio_schema_enumeration_reports_missing_scenario_graph_and_binding() -> None:
    missing = enumerate_studio_schema_registry(_schema_workspace(), "scenario:missing")
    assert any(issue.type == "missing_scenario" for issue in missing.issues)

    workspace = _schema_workspace()
    scenario = workspace.scenarios["scenario:train"]
    scenario.graph = None
    scenario.task_binding_spec = None
    registry = enumerate_studio_schema_registry(workspace, "scenario:train")

    issue_types = {issue.type for issue in registry.issues}
    assert "missing_graph" in issue_types
    assert "missing_task_binding_spec" in issue_types


def test_studio_schema_enumeration_validates_task_binding_schema_mismatch() -> None:
    workspace = _schema_workspace()
    scenario = next(iter(workspace.scenarios.values()))
    scenario.graph.nodes["network"].type = "Linear"
    assert scenario.task_binding_spec is not None
    scenario.task_binding_spec.exposed_data[0].dtype = "scalar"

    registry = enumerate_studio_schema_registry(workspace, scenario.id)
    issue_types = {issue.type for issue in registry.issues}

    assert "task_binding_dtype_mismatch" in issue_types


def test_studio_schema_enumeration_validates_task_binding_identity() -> None:
    workspace = _schema_workspace()
    scenario = next(iter(workspace.scenarios.values()))
    assert scenario.task_binding_spec is not None
    binding = scenario.task_binding_spec.bindings[0]
    scenario.task_binding_spec.bindings = [
        binding.model_copy(update={"id": "not-canonical"}),
        binding,
        binding.model_copy(),
    ]

    registry = enumerate_studio_schema_registry(workspace, scenario.id)
    issue_types = {issue.type for issue in registry.issues}

    assert "task_binding_id_mismatch" in issue_types
    assert "duplicate_task_binding" in issue_types


def test_studio_schema_enumerates_task_data_roles_and_rejects_protocol_bindings() -> None:
    workspace = _schema_workspace()
    scenario = workspace.scenarios["scenario:train"]
    assert scenario.task_binding_spec is not None
    scenario.task_binding_spec.exposed_data[1].bindable = True
    scenario.task_binding_spec.exposed_data[1].role = "target"
    scenario.task_binding_spec.bindings[0].source_data_id = "targets"

    registry = enumerate_studio_schema_registry(workspace, scenario.id)
    task_data = {item.path: item for item in registry.task_data}
    issue_types = {issue.type for issue in registry.issues}

    assert task_data["inputs"].role == "model_input"
    assert task_data["inputs"].bindable is True
    assert task_data["targets"].role == "target"
    assert task_data["targets"].bindable is False
    assert task_data["targets"].metadata["task_data_surface"] == "protocol"
    assert "task_data_bindable_role_mismatch" in issue_types
    assert "task_data_protocol_path_bindable" in issue_types
    assert "task_data_not_bindable" in issue_types


def test_studio_schema_accepts_component_parameter_bindings_with_declared_label() -> None:
    graph = GraphSpec(
        nodes={
            "field": {
                "type": "FixedField",
                "params": {
                    "scale": 1.0,
                    "amplitude": 1.0,
                    "field": [0.0, 0.0],
                    "active": False,
                    "label": "perturb",
                },
                "input_ports": ["force", "params_override"],
                "output_ports": ["force"],
            }
        },
        input_ports=["force"],
        output_ports=["force"],
        input_bindings={"force": ("field", "force")},
        output_bindings={"force": ("field", "force")},
    )
    task_binding = StudioTaskBindingSpec.model_validate(
        {
            "schema_version": "feedbax.spec.studio.task_bindings.v2",
            "exposed_data": [
                {
                    "id": "perturb",
                    "label": "Perturbation params",
                    "kind": "intervention",
                    "role": "component_parameter",
                    "path": "intervene.perturb",
                    "bindable": True,
                    "dtype": "object",
                    "value_spec": {
                        "mode": "constant",
                        "value": {"scale": 2.0, "active": True},
                    },
                    "metadata": {"temporal_support": "constant"},
                }
            ],
            "bindings": [
                {
                    "id": "task:perturb->field:params_override",
                    "source_data_id": "perturb",
                    "target_node_id": "field",
                    "target_port": "params_override",
                    "role": "component_parameter",
                    "metadata": {"task_parameter_label": "perturb"},
                }
            ],
            "metadata": {},
        }
    )

    issues = validate_task_binding_schema(task_binding, graph, "/task_binding_spec")
    assert not [issue for issue in issues if issue.severity == "error"]

    task_binding.bindings[0].metadata["task_parameter_label"] = "missing"
    issues = validate_task_binding_schema(task_binding, graph, "/task_binding_spec")
    assert {issue.type for issue in issues} >= {"component_parameter_label_unknown"}

    occupied_payload = graph.model_dump(mode="json", exclude_none=True)
    occupied_payload.update(
        {
            "nodes": {
                **occupied_payload["nodes"],
                "params_source": {
                    "type": "Constant",
                    "params": {"value": {"active": True}},
                    "input_ports": [],
                    "output_ports": ["output"],
                },
            },
            "wires": [
                {
                    "source_node": "params_source",
                    "source_port": "output",
                    "target_node": "field",
                    "target_port": "params_override",
                }
            ],
        }
    )
    occupied_graph = GraphSpec.model_validate(occupied_payload)
    task_binding.bindings[0].metadata["task_parameter_label"] = "perturb"
    issues = validate_task_binding_schema(task_binding, occupied_graph, "/task_binding_spec")
    assert {issue.type for issue in issues} >= {"task_binding_target_occupied"}


def test_studio_schema_enumeration_validates_intervention_targets() -> None:
    workspace = _schema_workspace()
    scenario = workspace.scenarios["scenario:train"]
    assert scenario.graph is not None
    scenario.graph.taps = [
        TapSpec.model_validate(item)
        for item in [
            {
                "id": "valid-clamp",
                "type": "intervention",
                "position": {"afterNode": "network"},
                "paths": {},
                "transform": {
                    "type": "intervention",
                    "params": {},
                    "intervention": {
                        "operation": "clamp",
                        "target_selector": {
                            "namespace": "state_path",
                            "compact": "path:states.net.output",
                            "path": "states.net.output",
                            "metadata": {},
                        },
                        "bounds": {"min": -1.0, "max": 1.0},
                        "metadata": {},
                    },
                },
            },
            {
                "id": "bad-constant",
                "type": "intervention",
                "position": {"afterNode": "network"},
                "paths": {},
                "transform": {
                    "type": "intervention",
                    "params": {},
                    "intervention": {
                        "operation": "constant",
                        "target_selector": {
                            "namespace": "task_data",
                            "compact": "task_data:targets",
                            "path": "targets",
                            "metadata": {},
                        },
                        "metadata": {},
                    },
                },
            },
        ]
    ]

    registry = enumerate_studio_schema_registry(workspace, "scenario:train")
    issue_types = {issue.type for issue in registry.issues}

    assert "intervention_missing_value" in issue_types
    assert "intervention_target_dtype_mismatch" in issue_types
    assert "intervention_missing_bounds" not in issue_types


def test_worker_emits_durable_training_manifest(
    tmp_path: Path,
    monkeypatch,
) -> None:
    monkeypatch.setenv("FEEDBAX_RUNS_DIR", str(tmp_path / "worker-runs"))
    event_queue: queue.Queue = queue.Queue()
    job = _Job(
        job_id="stub-job",
        total_batches=1,
        event_queue=event_queue,
        stop_event=threading.Event(),
        training_spec=_minimal_training_spec(),
        task_spec={"type": "SimpleReaches", "params": {}},
        task_binding_spec={
            "schema_version": "feedbax.spec.studio.task_bindings.v2",
            "exposed_data": [],
            "bindings": [],
            "metadata": {},
        },
        graph_spec=_minimal_graph_spec(),
        status=WorkerStatus.RUNNING,
    )

    job.status = WorkerStatus.COMPLETED
    job.last_loss = 0.5
    job.batch = 1
    _write_job_manifest(job)

    assert job.status == WorkerStatus.COMPLETED
    assert job.manifest_path is not None
    assert Path(job.manifest_path).exists()
    assert job.manifest_payload is not None
    assert job.manifest_payload["kind"] == "TrainingRunManifest"
    assert job.manifest_payload["artifacts"][0]["role"] == "training_history"

    events = []
    while not event_queue.empty():
        events.append(event_queue.get())
    log = next(event for event in events if event["type"] == "training_log")
    assert log["manifest_id"] == job.manifest_payload["id"]
    assert log["manifest_path"] == job.manifest_path


def _worker_contract_job(
    *,
    task_binding_spec: dict | None,
    graph_spec: dict | None = None,
) -> _Job:
    return _Job(
        job_id="worker-contract",
        total_batches=1,
        event_queue=queue.Queue(),
        stop_event=threading.Event(),
        training_spec=_minimal_training_spec(),
        task_spec={"type": "SimpleReaches", "params": {}},
        task_binding_spec=task_binding_spec,
        graph_spec=graph_spec or _minimal_graph_spec(),
        status=WorkerStatus.RUNNING,
    )


def test_worker_spec_contract_accepts_scenario_owned_task_binding_v2() -> None:
    _require_worker_specs(
        _worker_contract_job(
            task_binding_spec={
                "schema_version": "feedbax.spec.studio.task_bindings.v2",
                "exposed_data": [
                    {
                        "id": "inputs",
                        "label": "Inputs",
                        "kind": "signal",
                        "path": "inputs",
                        "bindable": True,
                        "metadata": {},
                    }
                ],
                "bindings": [
                    {
                        "id": "task:inputs->gain:input",
                        "source_data_id": "inputs",
                        "target_node_id": "gain",
                        "target_port": "input",
                        "role": "model_input",
                        "metadata": {},
                    }
                ],
                "metadata": {},
            }
        )
    )


def test_worker_spec_contract_migrates_legacy_task_binding_v1() -> None:
    job = _worker_contract_job(
        task_binding_spec={
            "schema_version": "feedbax.studio.task_bindings.v1",
            "exposed_outputs": [
                {
                    "id": "inputs",
                    "label": "Inputs",
                    "kind": "signal",
                    "path": "inputs",
                    "bindable": True,
                    "metadata": {},
                }
            ],
            "bindings": [
                {
                    "id": "task:inputs->gain:input",
                    "source_output_id": "inputs",
                    "target_node_id": "gain",
                    "target_port": "input",
                    "role": "model_input",
                    "metadata": {},
                }
            ],
            "metadata": {},
        }
    )

    _require_worker_specs(job)

    assert job.task_binding_spec["schema_version"] == "feedbax.spec.studio.task_bindings.v2"
    assert job.task_binding_spec["exposed_data"][0]["id"] == "inputs"
    assert "exposed_outputs" not in job.task_binding_spec
    assert job.task_binding_spec["bindings"][0]["source_data_id"] == "inputs"
    assert "source_output_id" not in job.task_binding_spec["bindings"][0]


def test_worker_spec_contract_rejects_runtime_network_payloads() -> None:
    job = _worker_contract_job(
        graph_spec=_runtime_network_graph_spec(),
        task_binding_spec={
            "schema_version": "feedbax.spec.studio.task_bindings.v2",
            "exposed_data": [
                {
                    "id": "inputs",
                    "label": "Inputs",
                    "kind": "signal",
                    "path": "inputs",
                    "bindable": True,
                    "metadata": {},
                }
            ],
            "bindings": [
                {
                    "id": "task:inputs->network:target",
                    "source_data_id": "inputs",
                    "target_node_id": "network",
                    "target_port": "target",
                    "role": "model_input",
                    "metadata": {},
                }
            ],
            "metadata": {},
        },
    )

    with pytest.raises(ValueError, match="task_binding_unknown_schema"):
        _require_worker_specs(job)


def test_worker_spec_contract_rejects_graph_incompatible_task_bindings() -> None:
    with pytest.raises(ValueError, match="unknown_task_binding_target_port"):
        _require_worker_specs(
            _worker_contract_job(
                task_binding_spec={
                    "schema_version": "feedbax.spec.studio.task_bindings.v2",
                    "exposed_data": [
                        {
                            "id": "inputs",
                            "label": "Inputs",
                            "kind": "signal",
                            "path": "inputs",
                            "bindable": True,
                            "metadata": {},
                        }
                    ],
                    "bindings": [
                        {
                            "id": "task:inputs->gain:missing",
                            "source_data_id": "inputs",
                            "target_node_id": "gain",
                            "target_port": "missing",
                            "role": "model_input",
                            "metadata": {},
                        }
                    ],
                    "metadata": {},
                }
            )
        )


@pytest.mark.parametrize(
    ("task_binding_spec", "message"),
    [
        (
            None,
            "must not be inferred from graph task nodes",
        ),
        (
            {
                "schema_version": "feedbax.spec.studio.task_bindings.v2",
                "exposed_outputs": [],
                "bindings": [],
                "metadata": {},
            },
            "exposed_outputs.*renamed to exposed_data",
        ),
        (
            {
                "schema_version": "feedbax.spec.studio.task_bindings.v2",
                "exposed_data": [],
                "bindings": [
                    {
                        "id": "task:inputs->network:input",
                        "source_output_id": "inputs",
                        "target_node_id": "network",
                        "target_port": "input",
                        "role": "model_input",
                        "metadata": {},
                    }
                ],
                "metadata": {},
            },
            "source_data_id",
        ),
    ],
)
def test_worker_spec_contract_rejects_legacy_or_inferred_task_bindings(
    task_binding_spec: dict | None,
    message: str,
) -> None:
    graph_with_task_node = _minimal_graph_spec()
    graph_with_task_node["nodes"]["task"] = {
        "type": "SimpleReaches",
        "params": {},
        "input_ports": [],
        "output_ports": ["inputs"],
    }
    with pytest.raises(ValueError, match=message):
        _require_worker_specs(
            _worker_contract_job(
                task_binding_spec=task_binding_spec,
                graph_spec=graph_with_task_node,
            )
        )


def test_worker_training_cfg_uses_task_n_steps() -> None:
    cfg = _extract_training_cfg(
        {"n_batches": 4, "n_reach_steps": 80},
        {"type": "DelayedReaches", "params": {"n_steps": 140}},
    )

    assert cfg.n_batches == 4
    assert cfg.n_reach_steps == 140


def test_worker_training_cfg_uses_timeline_task_n_steps() -> None:
    cfg = _extract_training_cfg(
        {"n_batches": 4, "n_reach_steps": 80},
        {"type": "DelayedReaches", "timeline": {"n_steps": 150}, "params": {"n_steps": 140}},
    )

    assert cfg.n_reach_steps == 150


def test_worker_training_cfg_parses_grad_clip_absent_null_and_float() -> None:
    absent_cfg = _extract_training_cfg({})
    null_cfg = _extract_training_cfg({"grad_clip": None})
    float_cfg = _extract_training_cfg({"grad_clip": "2.5"})

    assert absent_cfg.grad_clip == 1.0
    assert null_cfg.grad_clip is None
    assert float_cfg.grad_clip == 2.5


def test_worker_training_errors_instead_of_stub_on_missing_task_binding() -> None:
    event_queue: queue.Queue = queue.Queue()
    job = _Job(
        job_id="invalid-job",
        total_batches=1,
        event_queue=event_queue,
        stop_event=threading.Event(),
        training_spec=_minimal_training_spec(),
        task_spec={"type": "SimpleReaches", "params": {}},
        task_binding_spec=None,
        graph_spec=_minimal_graph_spec(),
        status=WorkerStatus.RUNNING,
    )

    _run_training(job)

    assert job.status == WorkerStatus.ERROR
    events = []
    while not event_queue.empty():
        events.append(event_queue.get())
    assert events[0]["type"] == "training_error"
    assert "task_binding_spec" in events[0]["error"]
    assert all(event is None or event.get("type") != "training_progress" for event in events)
