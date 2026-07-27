from __future__ import annotations

import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from feedbax.analysis import evaluation_compaction as compaction_module
from feedbax.analysis.evaluation_compaction import (
    EvaluationBatchConsumerInput,
    EvaluationBatchFragment,
    EvaluationBatchMergeInput,
    EvaluationBatchMergeState,
    compact_evaluation_batch,
    merge_evaluation_batch_fragment,
    publish_evaluation_compaction_products,
    reclaim_evaluation_batch_caches,
    register_evaluation_batch_consumer,
)
from feedbax.analysis.execution_context import EMPTY_STAGED_EXECUTION_CONTEXT
from feedbax.analysis.execution_context import (
    StagedCheckpointCustodyRootBinding,
    StagedExecutionContextError,
    resolve_staged_execution_context,
)
from feedbax.contracts.evaluation_lifecycle import (
    EvaluationBatchConsumerDeclaration,
    EvaluationLifecycleRowOutcome,
    EvaluationMatrixBatchPlan,
    EvaluationMatrixBatchUnit,
)
from feedbax.contracts.manifest import (
    AUTHENTICATED_MANIFEST_REF_SCHEMA_ID,
    AUTHENTICATED_MANIFEST_REF_SCHEMA_VERSION,
    AnalysisRunManifest,
    ParentRef,
    evaluation_states_cache_path,
    load_manifest,
    sha256_bytes,
)
from feedbax.contracts.staged_execution import (
    STAGED_EXECUTION_DESCRIPTOR_SCHEMA_ID,
    STAGED_EXECUTION_DESCRIPTOR_SCHEMA_VERSION,
    StagedCheckpointCustodySpec,
    StagedExecutionDescriptor,
)
from feedbax.persistence.artifact_custody import ImmutableArtifactBlobProvider


def _declaration(leaf_id: str) -> EvaluationBatchConsumerDeclaration:
    return EvaluationBatchConsumerDeclaration(
        leaf_id=leaf_id,
        consumer_id=f"tests.{leaf_id}",
        consumer_version="v1",
        accepted_evaluation_state_schema_ids=("tests.states.v1",),
        compact_product_schema_id=f"tests.{leaf_id}.product",
        compact_product_schema_version=f"tests.{leaf_id}.product.v1",
        compact_product_role=f"{leaf_id}_result",
        merge_state_schema_id=f"tests.{leaf_id}.merge",
        merge_state_schema_version=f"tests.{leaf_id}.merge.v1",
    )


def _register(declaration: EvaluationBatchConsumerDeclaration, calls: list[str]) -> None:
    def compact(value: EvaluationBatchConsumerInput) -> EvaluationBatchFragment:
        calls.append(f"compact:{declaration.leaf_id}:{value.batch.batch_id}")
        return EvaluationBatchFragment(
            payload={"rows": list(value.batch.ordered_row_ids)},
            schema_id=declaration.compact_product_schema_id,
            schema_version=declaration.compact_product_schema_version,
            role=declaration.compact_product_role,
        )

    def merge(value: EvaluationBatchMergeInput) -> EvaluationBatchMergeState:
        calls.append(f"merge:{declaration.leaf_id}:{value.batch.batch_id}")
        prior = [] if value.prior_merge_state is None else value.prior_merge_state["rows"]
        return EvaluationBatchMergeState(
            payload={"rows": [*prior, *value.fragment["rows"]]},
            schema_id=declaration.merge_state_schema_id,
            schema_version=declaration.merge_state_schema_version,
        )

    register_evaluation_batch_consumer(
        declaration.consumer_id,
        declaration.consumer_version,
        compact=compact,
        merge=merge,
        finalize=lambda value: EvaluationBatchFragment(
            payload=value.terminal_merge_state,
            schema_id=declaration.compact_product_schema_id,
            schema_version=declaration.compact_product_schema_version,
            role=declaration.compact_product_role,
        ),
        replace=True,
    )


def _batch_fixture(
    root: Path,
    batch: EvaluationMatrixBatchUnit,
) -> tuple[tuple[EvaluationLifecycleRowOutcome, ...], tuple[dict, ...]]:
    outcomes = []
    manifests = []
    for row_id in batch.ordered_row_ids:
        row_root = root / batch.batch_id / row_id
        manifest_id = f"evaluation:{row_id}"
        cache_path = evaluation_states_cache_path(manifest_id, root=row_root)
        cache_path.parent.mkdir(parents=True)
        cache_path.write_bytes(f"raw:{row_id}".encode())
        manifest_path = row_root / "manifest.json"
        manifest = {
            "id": manifest_id,
            "metadata": {"cache": {"states_path": str(cache_path)}},
        }
        manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
        outcomes.append(
            EvaluationLifecycleRowOutcome(
                row_id=row_id,
                manifest_id=manifest["id"],
                manifest_path=str(manifest_path),
                diagnostic_schema_ids=("tests.states.v1",),
            )
        )
        manifests.append(manifest)
    return tuple(outcomes), tuple(manifests)


def _parent_authorities(
    outcomes: tuple[EvaluationLifecycleRowOutcome, ...],
) -> tuple[ParentRef, ...]:
    authorities = []
    for outcome in outcomes:
        manifest_bytes = Path(outcome.manifest_path).read_bytes()
        authorities.append(
            ParentRef(
                kind="EvaluationRunManifest",
                id=outcome.manifest_id,
                role="evaluation_run",
                metadata={
                    "ref_schema_id": AUTHENTICATED_MANIFEST_REF_SCHEMA_ID,
                    "ref_schema_version": AUTHENTICATED_MANIFEST_REF_SCHEMA_VERSION,
                    "manifest_sha256": sha256_bytes(manifest_bytes),
                    "size_bytes": len(manifest_bytes),
                },
            )
        )
    return tuple(authorities)


def test_downstream_inventory_dag_covers_all_16200_rows_without_retained_authorities() -> None:
    leaves = tuple(
        _declaration(leaf) for leaf in ("trajectory", "command", "velocity", "controller", "pulse")
    )
    batches = (
        EvaluationMatrixBatchUnit(
            batch_id="nominal-target-zero",
            ordered_row_ids=tuple(f"nominal-zero-{index}" for index in range(110)),
            required_leaf_ids=("trajectory", "command", "velocity", "controller"),
        ),
        EvaluationMatrixBatchUnit(
            batch_id="nominal-other",
            ordered_row_ids=tuple(f"nominal-other-{index}" for index in range(7_810)),
            required_leaf_ids=("trajectory", "command", "velocity"),
        ),
        EvaluationMatrixBatchUnit(
            batch_id="hinf-1p05",
            ordered_row_ids=tuple(f"hinf-{index}" for index in range(360)),
            required_leaf_ids=("velocity",),
        ),
        EvaluationMatrixBatchUnit(
            batch_id="pulse",
            ordered_row_ids=tuple(f"pulse-{index}" for index in range(7_920)),
            required_leaf_ids=("pulse",),
        ),
    )
    plan = EvaluationMatrixBatchPlan(
        matrix_intent_hash="a" * 64,
        batches=batches,
        consumers=leaves,
    )

    assert sum(len(batch.ordered_row_ids) for batch in plan.batches) == 16_200
    assert {leaf for batch in plan.batches for leaf in batch.required_leaf_ids} == {
        "trajectory",
        "command",
        "velocity",
        "controller",
        "pulse",
    }
    assert "g1_induced_gain_projection_scalar" not in plan.model_dump_json()
    assert "s1_goal_hidden_map_radius_scalar" not in plan.model_dump_json()


def test_consumer_parameters_are_canonical_and_exactly_bound_to_callback_context(
    tmp_path: Path,
) -> None:
    declaration = _declaration("velocity").model_copy(
        update={"parameters": {"projection": "velocity", "window": 12}}
    )
    observed = []

    def compact(value: EvaluationBatchConsumerInput) -> EvaluationBatchFragment:
        observed.append((dict(value.parameters), value.execution_context))
        return EvaluationBatchFragment(
            payload={"projection": value.parameters["projection"]},
            schema_id=declaration.compact_product_schema_id,
            schema_version=declaration.compact_product_schema_version,
            role=declaration.compact_product_role,
        )

    register_evaluation_batch_consumer(
        declaration.consumer_id,
        declaration.consumer_version,
        compact=compact,
        merge=lambda _value: EvaluationBatchMergeState(
            payload={},
            schema_id=declaration.merge_state_schema_id,
            schema_version=declaration.merge_state_schema_version,
        ),
        finalize=lambda _value: EvaluationBatchFragment(
            payload={},
            schema_id=declaration.compact_product_schema_id,
            schema_version=declaration.compact_product_schema_version,
            role=declaration.compact_product_role,
        ),
        replace=True,
    )
    batch = EvaluationMatrixBatchUnit(
        batch_id="binding",
        ordered_row_ids=("row-a",),
        required_leaf_ids=("velocity",),
    )
    outcomes, manifests = _batch_fixture(tmp_path / "raw", batch)
    input_kwargs = {
        "matrix_intent_hash": "a" * 64,
        "batch": batch,
        "outcomes": outcomes,
        "manifests": manifests,
        "states": ({"value": 1},),
        "parent_authorities": (),
    }

    with pytest.raises(TypeError):
        EvaluationBatchConsumerInput(**input_kwargs)
    with pytest.raises(ValueError, match="resolved StagedExecutionContext"):
        compact_evaluation_batch(
            declaration,
            EvaluationBatchConsumerInput(
                **input_kwargs,
                parameters=declaration.parameters,
                execution_context=None,
            ),
            custody_root=tmp_path / "wrong-context",
        )
    with pytest.raises(ValueError, match="parameters drifted"):
        compact_evaluation_batch(
            declaration,
            EvaluationBatchConsumerInput(
                **input_kwargs,
                parameters={"projection": "position", "window": 12},
                execution_context=EMPTY_STAGED_EXECUTION_CONTEXT,
            ),
            custody_root=tmp_path / "drift",
        )
    fragment = compact_evaluation_batch(
        declaration,
        EvaluationBatchConsumerInput(
            **input_kwargs,
            parameters=declaration.parameters,
            execution_context=EMPTY_STAGED_EXECUTION_CONTEXT,
        ),
        custody_root=tmp_path / "custody",
    )

    assert observed == [
        (declaration.parameters, EMPTY_STAGED_EXECUTION_CONTEXT),
    ]
    assert fragment.metadata["consumer_parameters"] == declaration.parameters
    with pytest.raises(ValidationError):
        EvaluationBatchConsumerDeclaration.model_validate(
            {**declaration.model_dump(mode="json"), "parameters": {"path": tmp_path}}
        )
    with pytest.raises(ValidationError, match="canonical JSON"):
        EvaluationBatchConsumerDeclaration.model_validate(
            {**declaration.model_dump(mode="json"), "parameters": {"gain": float("nan")}}
        )


def test_consumer_callback_resolves_exact_authenticated_checkpoint_binding(
    tmp_path: Path,
) -> None:
    from tests.test_checkpoint_custody import (
        _resolver_parent_ref,
        _write_resolver_checkpoint,
    )

    checkpoint_root = tmp_path / "checkpoint-custody"
    checkpoint = _write_resolver_checkpoint(checkpoint_root)
    parent = _resolver_parent_ref(checkpoint)
    parent = parent.model_copy(
        update={
            "metadata": {
                **parent.metadata,
                "checkpoint_custody_binding": "checkpoints",
            }
        }
    )
    descriptor = StagedExecutionDescriptor(
        schema_id=STAGED_EXECUTION_DESCRIPTOR_SCHEMA_ID,
        schema_version=STAGED_EXECUTION_DESCRIPTOR_SCHEMA_VERSION,
        artifact_providers={},
        checkpoint_custody={
            "checkpoints": StagedCheckpointCustodySpec(
                backend="feedbax-checkpoint-transaction-tree"
            )
        },
    )
    context = resolve_staged_execution_context(
        descriptor,
        checkpoint_custody_bindings=[
            StagedCheckpointCustodyRootBinding("checkpoints", checkpoint_root)
        ],
    )
    declaration = _declaration("velocity").model_copy(
        update={
            "parameters": {
                "checkpoint_binding": "checkpoints",
                "checkpoint_ref": parent.model_dump(mode="json"),
            }
        }
    )
    observed_contexts = []

    def compact(value: EvaluationBatchConsumerInput) -> EvaluationBatchFragment:
        observed_contexts.append(value.execution_context)
        ref = ParentRef.model_validate(value.parameters["checkpoint_ref"])
        resolved = value.execution_context.resolve_checkpoint_custody_ref(
            ref,
            binding_name=str(value.parameters["checkpoint_binding"]),
            slot_names=["controller"],
        )
        assert resolved.parent_ref == ref
        assert set(resolved.slots) == {"controller"}
        return EvaluationBatchFragment(
            payload={"transaction_id": resolved.manifest.transaction_id},
            schema_id=declaration.compact_product_schema_id,
            schema_version=declaration.compact_product_schema_version,
            role=declaration.compact_product_role,
        )

    register_evaluation_batch_consumer(
        declaration.consumer_id,
        declaration.consumer_version,
        compact=compact,
        merge=lambda _value: EvaluationBatchMergeState(
            payload={},
            schema_id=declaration.merge_state_schema_id,
            schema_version=declaration.merge_state_schema_version,
        ),
        finalize=lambda _value: EvaluationBatchFragment(
            payload={},
            schema_id=declaration.compact_product_schema_id,
            schema_version=declaration.compact_product_schema_version,
            role=declaration.compact_product_role,
        ),
        replace=True,
    )
    batch = EvaluationMatrixBatchUnit(
        batch_id="checkpoint-binding",
        ordered_row_ids=("row-a",),
        required_leaf_ids=("velocity",),
    )
    outcomes, manifests = _batch_fixture(tmp_path / "raw", batch)

    def consumer_input(
        configured: EvaluationBatchConsumerDeclaration,
        execution_context,
    ) -> EvaluationBatchConsumerInput:
        return EvaluationBatchConsumerInput(
            matrix_intent_hash="a" * 64,
            batch=batch,
            outcomes=outcomes,
            manifests=manifests,
            states=({"value": 1},),
            parent_authorities=(),
            parameters=configured.parameters,
            execution_context=execution_context,
        )

    fragment = compact_evaluation_batch(
        declaration,
        consumer_input(declaration, context),
        custody_root=tmp_path / "published",
    )
    assert observed_contexts == [context]
    assert fragment.metadata["consumer_parameters"] == declaration.parameters

    for binding, execution_context in (
        ("wrong", context),
        ("checkpoints", EMPTY_STAGED_EXECUTION_CONTEXT),
    ):
        configured = declaration.model_copy(
            update={
                "parameters": {
                    **declaration.parameters,
                    "checkpoint_binding": binding,
                }
            }
        )
        custody_root = tmp_path / f"rejected-{binding}-{len(observed_contexts)}"
        with pytest.raises(StagedExecutionContextError):
            compact_evaluation_batch(
                configured,
                consumer_input(configured, execution_context),
                custody_root=custody_root,
            )
        assert not custody_root.exists()
    cache_path = Path(manifests[0]["metadata"]["cache"]["states_path"])
    assert cache_path.is_file()


def test_ordered_merge_waits_for_every_leaf_then_reclaims_and_publishes(tmp_path: Path) -> None:
    declarations = (_declaration("trajectory"), _declaration("velocity"))
    calls: list[str] = []
    for declaration in declarations:
        _register(declaration, calls)
    batches = (
        EvaluationMatrixBatchUnit(
            batch_id="0000",
            ordered_row_ids=("row-a",),
            required_leaf_ids=("trajectory", "velocity"),
        ),
        EvaluationMatrixBatchUnit(
            batch_id="0001",
            ordered_row_ids=("row-b",),
            required_leaf_ids=("trajectory", "velocity"),
        ),
    )
    prior = {}
    all_outcomes = []
    for batch_index, batch in enumerate(batches):
        outcomes, manifests = _batch_fixture(tmp_path / "raw", batch)
        all_outcomes.extend(outcomes)
        fragments = [
            compact_evaluation_batch(
                declaration,
                EvaluationBatchConsumerInput(
                    matrix_intent_hash="a" * 64,
                    batch=batch,
                    outcomes=outcomes,
                    manifests=manifests,
                    states=({"row": batch.ordered_row_ids[0]},),
                    parent_authorities=(),
                    parameters=declaration.parameters,
                    execution_context=EMPTY_STAGED_EXECUTION_CONTEXT,
                ),
                custody_root=tmp_path / "custody",
            )
            for declaration in declarations
        ]
        acknowledgements = [
            merge_evaluation_batch_fragment(
                declaration,
                matrix_intent_hash="a" * 64,
                batch=batch,
                parent_authorities=_parent_authorities(outcomes),
                fragment=fragment,
                prior_merge_state=prior.get(declaration.leaf_id),
                custody_root=tmp_path / "custody",
                execution_context=EMPTY_STAGED_EXECUTION_CONTEXT,
            )
            for declaration, fragment in zip(declarations, fragments, strict=True)
        ]
        for acknowledgement in acknowledgements:
            prior[acknowledgement.leaf_id] = acknowledgement.merge_state
        cache_path = Path(
            json.loads(Path(outcomes[0].manifest_path).read_text())["metadata"]["cache"][
                "states_path"
            ]
        )
        with pytest.raises(ValueError, match="every declared leaf"):
            reclaim_evaluation_batch_caches(
                batch,
                batch_index=batch_index,
                outcomes=outcomes,
                acknowledgements=acknowledgements[:1],
                required_declarations=declarations,
                custody_root=tmp_path / "custody",
                execution_context=EMPTY_STAGED_EXECUTION_CONTEXT,
            )
        assert cache_path.exists()
        reclaim_evaluation_batch_caches(
            batch,
            batch_index=batch_index,
            outcomes=outcomes,
            acknowledgements=acknowledgements,
            required_declarations=declarations,
            custody_root=tmp_path / "custody",
            execution_context=EMPTY_STAGED_EXECUTION_CONTEXT,
        )
        assert not cache_path.exists()
        if batch_index == 0:
            reclamation_checkpoint = tmp_path / "custody" / "reclamation-checkpoints" / "0000.json"
            legacy_reclamation = json.loads(reclamation_checkpoint.read_text())
            for acknowledgement in legacy_reclamation["leaf_acknowledgements"]:
                acknowledgement.pop("parameters")
                for artifact_name in ("fragment", "merge_state"):
                    metadata = acknowledgement[artifact_name]["metadata"]
                    metadata.pop("consumer_parameters")
                    metadata.pop("consumer_parameters_sha256")
            reclamation_checkpoint.write_text(
                json.dumps(legacy_reclamation),
                encoding="utf-8",
            )
            legacy_resume = reclaim_evaluation_batch_caches(
                batch,
                batch_index=batch_index,
                outcomes=outcomes,
                acknowledgements=acknowledgements,
                required_declarations=declarations,
                custody_root=tmp_path / "custody",
                execution_context=EMPTY_STAGED_EXECUTION_CONTEXT,
            )
            assert legacy_resume.removed_cache_manifest_ids == ("evaluation:row-a",)
            reclamation_checkpoint.unlink()
            intent_path = tmp_path / "custody" / "reclamation-intents" / "0000.json"
            intent = json.loads(intent_path.read_text())
            intent["entries"][0]["status"] = "deleting"
            intent_path.write_text(json.dumps(intent), encoding="utf-8")
            resumed_reclamation = reclaim_evaluation_batch_caches(
                batch,
                batch_index=batch_index,
                outcomes=outcomes,
                acknowledgements=acknowledgements,
                required_declarations=declarations,
                custody_root=tmp_path / "custody",
                execution_context=EMPTY_STAGED_EXECUTION_CONTEXT,
            )
            assert resumed_reclamation.removed_cache_manifest_ids == ("evaluation:row-a",)

    terminal = publish_evaluation_compaction_products(
        declarations,
        prior,
        all_outcomes,
        custody_root=tmp_path / "custody",
        execution_context=EMPTY_STAGED_EXECUTION_CONTEXT,
    )[0]
    manifest_bytes = ImmutableArtifactBlobProvider(tmp_path / "custody").get_bytes(terminal)
    manifest_path = tmp_path / "terminal.json"
    manifest_path.write_bytes(manifest_bytes)
    manifest = load_manifest(manifest_path)
    assert isinstance(manifest, AnalysisRunManifest)
    assert [item.role for item in manifest.produced_data] == [
        "trajectory_result",
        "velocity_result",
    ]
    assert calls == [
        "compact:trajectory:0000",
        "compact:velocity:0000",
        "merge:trajectory:0000",
        "merge:velocity:0000",
        "compact:trajectory:0001",
        "compact:velocity:0001",
        "merge:trajectory:0001",
        "merge:velocity:0001",
    ]


def test_resume_reuses_verified_merge_checkpoint_without_double_application(tmp_path: Path) -> None:
    declaration = _declaration("velocity")
    calls: list[str] = []
    _register(declaration, calls)
    batch = EvaluationMatrixBatchUnit(
        batch_id="resume",
        ordered_row_ids=("row-a",),
        required_leaf_ids=("velocity",),
    )
    outcomes, manifests = _batch_fixture(tmp_path / "raw", batch)
    fragment = compact_evaluation_batch(
        declaration,
        EvaluationBatchConsumerInput(
            matrix_intent_hash="a" * 64,
            batch=batch,
            outcomes=outcomes,
            manifests=manifests,
            states=({"value": 1},),
            parent_authorities=(),
            parameters=declaration.parameters,
            execution_context=EMPTY_STAGED_EXECUTION_CONTEXT,
        ),
        custody_root=tmp_path / "custody",
    )
    first = merge_evaluation_batch_fragment(
        declaration,
        matrix_intent_hash="a" * 64,
        batch=batch,
        parent_authorities=_parent_authorities(outcomes),
        fragment=fragment,
        prior_merge_state=None,
        custody_root=tmp_path / "custody",
        execution_context=EMPTY_STAGED_EXECUTION_CONTEXT,
    )
    checkpoint_path = tmp_path / "custody" / "merge-checkpoints" / "velocity" / "resume.json"
    legacy_checkpoint = json.loads(checkpoint_path.read_text())
    legacy_checkpoint["schema_version"] = (
        "feedbax.orchestration.evaluation_batch_merge_checkpoint.v1"
    )
    legacy_checkpoint["declaration"].pop("parameters")
    legacy_checkpoint["acknowledgement"].pop("parameters")
    for artifact_name in ("fragment", "merge_state"):
        metadata = legacy_checkpoint["acknowledgement"][artifact_name]["metadata"]
        metadata.pop("consumer_parameters")
        metadata.pop("consumer_parameters_sha256")
    checkpoint_path.write_text(json.dumps(legacy_checkpoint), encoding="utf-8")
    resumed = merge_evaluation_batch_fragment(
        declaration,
        matrix_intent_hash="a" * 64,
        batch=batch,
        parent_authorities=_parent_authorities(outcomes),
        fragment=fragment,
        prior_merge_state=None,
        custody_root=tmp_path / "custody",
        execution_context=EMPTY_STAGED_EXECUTION_CONTEXT,
    )

    assert first.merge_state.sha256 == resumed.merge_state.sha256
    assert resumed.reused_verified_fragment is True
    assert calls.count("merge:velocity:resume") == 1
    changed = declaration.model_copy(update={"compact_product_role": "changed_role"})
    with pytest.raises(ValueError, match="fragment identity drifted"):
        merge_evaluation_batch_fragment(
            changed,
            matrix_intent_hash="a" * 64,
            batch=batch,
            parent_authorities=_parent_authorities(outcomes),
            fragment=fragment,
            prior_merge_state=None,
            custody_root=tmp_path / "custody",
            execution_context=EMPTY_STAGED_EXECUTION_CONTEXT,
        )
    changed_parameters = declaration.model_copy(update={"parameters": {"projection": "position"}})
    with pytest.raises(ValueError, match="fragment identity drifted"):
        merge_evaluation_batch_fragment(
            changed_parameters,
            matrix_intent_hash="a" * 64,
            batch=batch,
            parent_authorities=_parent_authorities(outcomes),
            fragment=fragment,
            prior_merge_state=None,
            custody_root=tmp_path / "custody",
            execution_context=EMPTY_STAGED_EXECUTION_CONTEXT,
        )
    changed_batch = batch.model_copy(update={"ordered_row_ids": ("row-b",)})
    with pytest.raises(ValueError, match="checkpoint identity drifted"):
        merge_evaluation_batch_fragment(
            declaration,
            matrix_intent_hash="a" * 64,
            batch=changed_batch,
            parent_authorities=_parent_authorities(outcomes),
            fragment=fragment,
            prior_merge_state=None,
            custody_root=tmp_path / "custody",
            execution_context=EMPTY_STAGED_EXECUTION_CONTEXT,
        )
    changed_parent = _parent_authorities(outcomes)[0].model_copy(
        update={
            "metadata": {
                **_parent_authorities(outcomes)[0].metadata,
                "manifest_sha256": "b" * 64,
            }
        }
    )
    with pytest.raises(ValueError, match="checkpoint identity drifted"):
        merge_evaluation_batch_fragment(
            declaration,
            matrix_intent_hash="a" * 64,
            batch=batch,
            parent_authorities=(changed_parent,),
            fragment=fragment,
            prior_merge_state=None,
            custody_root=tmp_path / "custody",
            execution_context=EMPTY_STAGED_EXECUTION_CONTEXT,
        )


def test_reclamation_restart_rejects_canonical_numeric_parameter_drift_before_delete(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    declaration = _declaration("velocity").model_copy(update={"parameters": {"threshold": 1}})
    calls: list[str] = []
    _register(declaration, calls)
    batch = EvaluationMatrixBatchUnit(
        batch_id="numeric-drift",
        ordered_row_ids=("row-a",),
        required_leaf_ids=("velocity",),
    )
    outcomes, manifests = _batch_fixture(tmp_path / "raw", batch)
    fragment = compact_evaluation_batch(
        declaration,
        EvaluationBatchConsumerInput(
            matrix_intent_hash="a" * 64,
            batch=batch,
            outcomes=outcomes,
            manifests=manifests,
            states=({"value": 1},),
            parent_authorities=(),
            parameters=declaration.parameters,
            execution_context=EMPTY_STAGED_EXECUTION_CONTEXT,
        ),
        custody_root=tmp_path / "custody",
    )
    acknowledgement = merge_evaluation_batch_fragment(
        declaration,
        matrix_intent_hash="a" * 64,
        batch=batch,
        parent_authorities=_parent_authorities(outcomes),
        fragment=fragment,
        prior_merge_state=None,
        custody_root=tmp_path / "custody",
        execution_context=EMPTY_STAGED_EXECUTION_CONTEXT,
    )
    cache_path = Path(manifests[0]["metadata"]["cache"]["states_path"])
    real_atomic_write = compaction_module._atomic_write_json

    def interrupt_after_intent(path: Path, payload: dict) -> None:
        real_atomic_write(path, payload)
        if (
            path.parent.name == "reclamation-intents"
            and payload["entries"][0]["status"] == "pending"
        ):
            raise RuntimeError("simulated interruption before deletion")

    monkeypatch.setattr(
        compaction_module,
        "_atomic_write_json",
        interrupt_after_intent,
    )
    with pytest.raises(RuntimeError, match="before deletion"):
        reclaim_evaluation_batch_caches(
            batch,
            batch_index=0,
            outcomes=outcomes,
            acknowledgements=(acknowledgement,),
            required_declarations=(declaration,),
            custody_root=tmp_path / "custody",
            execution_context=EMPTY_STAGED_EXECUTION_CONTEXT,
        )
    assert cache_path.is_file()
    monkeypatch.setattr(compaction_module, "_atomic_write_json", real_atomic_write)

    drifted_declaration = declaration.model_copy(update={"parameters": {"threshold": 1.0}})
    drifted_acknowledgement = acknowledgement.model_copy(update={"parameters": {"threshold": 1.0}})
    with pytest.raises(ValueError, match="reclamation intent identity drifted"):
        reclaim_evaluation_batch_caches(
            batch,
            batch_index=0,
            outcomes=outcomes,
            acknowledgements=(drifted_acknowledgement,),
            required_declarations=(drifted_declaration,),
            custody_root=tmp_path / "custody",
            execution_context=EMPTY_STAGED_EXECUTION_CONTEXT,
        )
    assert cache_path.is_file()


@pytest.mark.parametrize("failure", ["wrong_schema", "wrong_role", "tampered", "unmaterialized"])
def test_fragment_failure_retains_raw_cache(tmp_path: Path, failure: str) -> None:
    declaration = _declaration("velocity")

    def compact(_value: EvaluationBatchConsumerInput) -> EvaluationBatchFragment:
        return EvaluationBatchFragment(
            payload={"rows": ["row-a"]},
            schema_id=(
                "tests.wrong"
                if failure == "wrong_schema"
                else declaration.compact_product_schema_id
            ),
            schema_version=declaration.compact_product_schema_version,
            role="wrong" if failure == "wrong_role" else declaration.compact_product_role,
        )

    register_evaluation_batch_consumer(
        declaration.consumer_id,
        declaration.consumer_version,
        compact=compact,
        merge=lambda _value: EvaluationBatchMergeState(
            payload={"rows": ["row-a"]},
            schema_id=declaration.merge_state_schema_id,
            schema_version=declaration.merge_state_schema_version,
        ),
        finalize=lambda value: EvaluationBatchFragment(
            payload=value.terminal_merge_state,
            schema_id=declaration.compact_product_schema_id,
            schema_version=declaration.compact_product_schema_version,
            role=declaration.compact_product_role,
        ),
        replace=True,
    )
    batch = EvaluationMatrixBatchUnit(
        batch_id="failure",
        ordered_row_ids=("row-a",),
        required_leaf_ids=("velocity",),
    )
    outcomes, manifests = _batch_fixture(tmp_path / "raw", batch)
    cache_path = Path(manifests[0]["metadata"]["cache"]["states_path"])
    if failure in {"wrong_schema", "wrong_role"}:
        with pytest.raises(ValueError, match="wrong fragment contract"):
            compact_evaluation_batch(
                declaration,
                EvaluationBatchConsumerInput(
                    matrix_intent_hash="a" * 64,
                    batch=batch,
                    outcomes=outcomes,
                    manifests=manifests,
                    states=({"value": 1},),
                    parent_authorities=(),
                    parameters=declaration.parameters,
                    execution_context=EMPTY_STAGED_EXECUTION_CONTEXT,
                ),
                custody_root=tmp_path / "custody",
            )
    else:
        fragment = compact_evaluation_batch(
            declaration,
            EvaluationBatchConsumerInput(
                matrix_intent_hash="a" * 64,
                batch=batch,
                outcomes=outcomes,
                manifests=manifests,
                states=({"value": 1},),
                parent_authorities=(),
                parameters=declaration.parameters,
                execution_context=EMPTY_STAGED_EXECUTION_CONTEXT,
            ),
            custody_root=tmp_path / "custody",
        )
        provider = ImmutableArtifactBlobProvider(tmp_path / "custody")
        fragment_path = tmp_path / "custody" / provider.canonical_relative_path(fragment)
        if failure == "tampered":
            fragment_path.write_bytes(b"tampered")
        else:
            fragment_path.unlink()
        with pytest.raises(Exception):
            merge_evaluation_batch_fragment(
                declaration,
                matrix_intent_hash="a" * 64,
                batch=batch,
                parent_authorities=_parent_authorities(outcomes),
                fragment=fragment,
                prior_merge_state=None,
                custody_root=tmp_path / "custody",
                execution_context=EMPTY_STAGED_EXECUTION_CONTEXT,
            )
    assert cache_path.exists()
