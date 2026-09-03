from __future__ import annotations

from copy import deepcopy

import jax.numpy as jnp
import jax.random as jr
import pytest
from pydantic import ValidationError

from feedbax.contracts.graph import (
    GraphSpec,
    StudioPersistenceDocument,
    StudioTaskBindingSpec,
    StudioTaskTimelineSpec,
    StudioWorkspaceSpec,
    studio_semantic_document_sha256,
)
from feedbax.contracts.migrations import default_spec_registry
from feedbax.contracts.training import TaskSpec
from feedbax.web.services.graph_service import GraphService
from feedbax.web.worker.execution import _materialize_task_data, _validate_epoch_value_targets


def _literal(value: object) -> dict[str, object]:
    return {
        "schema_version": "feedbax.spec.studio.value.v2",
        "value_form": "literal",
        "variation": {"scope": "fixed", "metadata": {}},
        "mode": "constant",
        "value": value,
        "metadata": {},
    }


def _timeline(*, first: float = 1.0, second: float = 3.0) -> dict[str, object]:
    return {
        "schema_id": "feedbax.spec.studio.task_timeline",
        "schema_version": "feedbax.spec.studio.task_timeline.v2",
        "epochs": [
            {"id": "prep", "label": "Prep", "index": 0, "length": _literal(2)},
            {"id": "move", "label": "Move", "index": 1, "length": _literal(None)},
        ],
        "signals": [
            {
                "id": "hold",
                "label": "Hold",
                "kind": "signal",
                "task_data_id": "hold",
                "path": "inputs.hold",
                "value_spec": _literal(0),
                "metadata": {},
            }
        ],
        "epoch_value_specs": [
            {"target_id": "hold", "epoch_id": "prep", "value_spec": _literal(first)},
            {"target_id": "hold", "epoch_id": "move", "value_spec": _literal(second)},
        ],
        "segments": [],
        "metadata": {},
    }


def _binding() -> StudioTaskBindingSpec:
    return StudioTaskBindingSpec.model_validate(
        {
            "schema_version": "feedbax.spec.studio.task_bindings.v2",
            "exposed_data": [
                {
                    "id": "hold",
                    "label": "Hold",
                    "kind": "signal",
                    "path": "inputs.hold",
                    "bindable": True,
                    "expected_shape": ["time", 1],
                    "value_spec": _literal(0),
                    "metadata": {},
                }
            ],
            "bindings": [],
            "metadata": {},
        }
    )


def _workspace(timeline: dict[str, object]) -> StudioWorkspaceSpec:
    return StudioWorkspaceSpec.model_validate(
        {
            "id": "workspace:test",
            "label": "Timeline test",
            "stages": [],
            "scenarios": {
                "scenario:train": {
                    "id": "scenario:train",
                    "label": "Train",
                    "task_spec": {"type": "DelayedReaches", "params": {}, "timeline": timeline},
                }
            },
        }
    )


def test_v1_migrates_and_current_round_trips() -> None:
    migrated = StudioTaskTimelineSpec.model_validate(
        {
            "schema_version": "feedbax.spec.studio.task_timeline.v1",
            "epochs": [{"id": "prep", "label": "Prep", "index": 0, "length": _literal(2)}],
            "signals": [],
        }
    )
    assert migrated.schema_version == "feedbax.spec.studio.task_timeline.v2"
    assert migrated.epoch_value_specs == []
    assert default_spec_registry.current_version("StudioTaskTimelineSpec") == (
        "feedbax.spec.studio.task_timeline.v2"
    )
    assert StudioTaskTimelineSpec.model_validate_json(migrated.model_dump_json()) == migrated


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (
            lambda value: value.update(schema_version="feedbax.spec.studio.task_timeline.v99"),
            "unsupported StudioTaskTimelineSpec schema_version",
        ),
        (
            lambda value: value["epoch_value_specs"].append(
                {"target_id": "hold", "epoch_id": "prep", "value_spec": _literal(2)}
            ),
            "overlapping timeline epoch values",
        ),
        (
            lambda value: value["epoch_value_specs"][0].update(target_id="missing"),
            "unknown timeline epoch-value target",
        ),
        (
            lambda value: value["epoch_value_specs"][0]["value_spec"].update(value=float("nan")),
            "non-finite number",
        ),
        (
            lambda value: value["epoch_value_specs"][0].update(
                value_spec={
                    **_literal(None),
                    "value_form": "distribution",
                    "mode": "distribution",
                    "distribution": {"family": "uniform", "parameters": {"min": 0}},
                }
            ),
            "numeric uniform min/max",
        ),
    ],
)
def test_malformed_future_overlap_unknown_and_unsafe_values_fail_closed(
    mutate,
    message: str,
) -> None:
    payload = _timeline()
    mutate(payload)
    with pytest.raises(ValidationError, match=message):
        StudioTaskTimelineSpec.model_validate(payload)


def test_timeline_semantics_change_identity_while_layout_is_excluded() -> None:
    graph = GraphSpec()
    first = _workspace(_timeline(first=1.0))
    changed = _workspace(_timeline(first=2.0))
    identity = studio_semantic_document_sha256(graph, first)
    assert studio_semantic_document_sha256(graph, changed) != identity
    layout_only = {"analysis_canvas_layout": {"stages": {"train": {"pages": {}}}}}
    assert layout_only
    assert studio_semantic_document_sha256(graph, first) == identity


def test_admission_failure_writes_nothing(tmp_path) -> None:
    service = GraphService(storage_dir=tmp_path)
    bad_workspace = _workspace(_timeline()).model_dump(mode="json")
    bad_workspace["scenarios"]["scenario:train"]["task_spec"]["timeline"]["schema_version"] = (
        "feedbax.spec.studio.task_timeline.v99"
    )
    with pytest.raises(ValidationError):
        StudioPersistenceDocument.model_validate(
            {
                "schema_id": "feedbax.spec.studio.persistence_document",
                "schema_version": "feedbax.spec.studio.persistence_document.v1",
                "graph": GraphSpec().model_dump(mode="json"),
                "workspace": bad_workspace,
            }
        )
    assert list(tmp_path.iterdir()) == []
    assert service.list_graphs() == []


def test_save_load_preserves_exact_epoch_values(tmp_path) -> None:
    service = GraphService(storage_dir=tmp_path)
    record = service.create_graph(GraphSpec(), workspace=_workspace(_timeline()))
    loaded = service.get_graph(record.graph_id).project.workspace
    assert loaded is not None
    timeline = loaded.scenarios["scenario:train"].task_spec["timeline"]
    assert timeline["epoch_value_specs"][0]["value_spec"]["value"] == 1.0
    assert timeline["epoch_value_specs"][1]["value_spec"]["value"] == 3.0


def test_execution_applies_distinct_values_at_exact_epoch_boundary() -> None:
    task = TaskSpec.model_validate(
        {"type": "DelayedReaches", "params": {}, "timeline": _timeline()}
    )
    data = _materialize_task_data(_binding(), task, 5, key=jr.PRNGKey(0))
    assert jnp.array_equal(data["hold"][:, 0], jnp.asarray([1, 1, 3, 3, 3]))


def test_execution_rejects_authored_target_missing_from_runtime_binding() -> None:
    task = TaskSpec.model_validate(
        {"type": "DelayedReaches", "params": {}, "timeline": _timeline()}
    )
    binding = _binding().model_copy(
        update={"exposed_data": []},
    )
    with pytest.raises(ValueError, match="not exposed by task_binding_spec"):
        _validate_epoch_value_targets(task.timeline, binding)


def test_equivalent_reordering_is_rejected_instead_of_changing_identity() -> None:
    payload = deepcopy(_timeline())
    payload["epoch_value_specs"].reverse()
    with pytest.raises(ValidationError, match="ordered by target_id then epoch index"):
        StudioTaskTimelineSpec.model_validate(payload)
