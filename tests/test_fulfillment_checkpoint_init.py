"""Checkpoint initialization, from a compile lock to an authenticated request.

Everything here is stated over ``quillon``'s compiled outputs and a real
checkpoint custody tree written by
:func:`feedbax.training.checkpoint_custody.write_checkpoint_transaction`. Three
claims are under test:

* **the request follows from the lock's typed reference**, and from nothing a
  compiled document claims: the row, the mode, and the source identity are the
  binding's and the reference's;
* **the source is authenticated against real bytes** under a custody root the
  caller explicitly binds, and lowering runs the closed matching rule against the
  structure those bytes declare;
* **an absent, moved, or altered checkpoint is a refusal**, never a silently
  fresh start, and a reference that cannot authenticate a source refuses before
  anything is resolved.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from feedbax.analysis.execution_context import (
    StagedCheckpointCustodyRootBinding,
    StagedExecutionContextError,
    resolve_staged_execution_context,
)
from feedbax.analysis.fulfillment_checkpoint_init import (
    CHECKPOINT_CUSTODY_ROLE,
    CheckpointInitializationWiringError,
    authenticate_checkpoint_initialization_source,
    checkpoint_initialization_requests,
    checkpoint_initialization_source_structure,
    compiled_checkpoint_initializations,
    experiment_checkpoint_initializations,
    lower_planned_checkpoint_initialization,
)
from feedbax.analysis.fulfillment_derivation import (
    derive_fulfillment_plan,
    read_compiled_outputs,
)
from feedbax.contracts.experiment_compile_lock import (
    AuthenticatedReceiptReference,
    CheckpointInitializationBinding,
    ReceiptLocatorReference,
)
from feedbax.contracts.manifest import sha256_bytes
from feedbax.contracts.staged_execution import (
    STAGED_EXECUTION_DESCRIPTOR_SCHEMA_ID,
    STAGED_EXECUTION_DESCRIPTOR_SCHEMA_VERSION,
    StagedCheckpointCustodySpec,
    StagedExecutionDescriptor,
)
from feedbax.training.checkpoint_custody import write_checkpoint_transaction

from tests.fake_project_experiment.products import QuillonOutputs
from tests.test_checkpoint_custody import _coordinate, _minimax_slots, _run_spec

CUSTODY_BINDING = "training-checkpoints"
ROW = "cohort-row-0"


@pytest.fixture
def outputs(tmp_path: Path) -> QuillonOutputs:
    return QuillonOutputs(tmp_path / "repo")


@pytest.fixture
def custody_root(tmp_path: Path) -> Path:
    root = tmp_path / "checkpoints"
    root.mkdir()
    return root


def _write_checkpoint(root: Path):
    """Write one real checkpoint transaction into a custody root."""
    run_spec = _run_spec(minimax=True)
    return write_checkpoint_transaction(
        root,
        run_spec=run_spec,
        phase_program=run_spec.worker_execution.method_contract.phase_program,
        barrier_name="after_warmup",
        coordinate=_coordinate(),
        slots=_minimax_slots(),
    )


def _reference(result, root: Path, *, mode: str = "continue_from", row: str = ROW, **changes):
    """The authenticated reference a compile emits for one checkpoint source."""
    fields = {
        "manifest_kind": "TrainingCheckpointTransactionManifest",
        "manifest_id": result.manifest.transaction_id,
        "manifest_sha256": sha256_bytes(result.manifest_path.read_bytes()),
        "size_bytes": len(result.manifest_path.read_bytes()),
        "role_path": f"rows.{row}.checkpoint_initialization",
        "consumer": CheckpointInitializationBinding(mode=mode, row_id=row),
        "execution_uri": result.manifest_path.relative_to(root).as_posix(),
    }
    fields.update(changes)
    return AuthenticatedReceiptReference(**fields)


def _context(root: Path):
    return resolve_staged_execution_context(
        StagedExecutionDescriptor(
            schema_id=STAGED_EXECUTION_DESCRIPTOR_SCHEMA_ID,
            schema_version=STAGED_EXECUTION_DESCRIPTOR_SCHEMA_VERSION,
            artifact_providers={},
            checkpoint_custody={
                CUSTODY_BINDING: StagedCheckpointCustodySpec(
                    backend="feedbax-checkpoint-transaction-tree"
                )
            },
        ),
        checkpoint_custody_bindings=(
            StagedCheckpointCustodyRootBinding(CUSTODY_BINDING, root),
        ),
    )


def _entries(outputs: QuillonOutputs, name: str, references):
    """Derive one training target's plan and read its checkpoint initializations."""
    outputs.cohort(name, references=references)
    index = read_compiled_outputs(outputs.output_directory)
    return checkpoint_initialization_requests(
        derive_fulfillment_plan(index, target=name), index=index
    )


# --------------------------------------------------------------------------
# Planning: the request follows from the lock's typed reference
# --------------------------------------------------------------------------


def test_a_lock_reference_becomes_one_authenticated_request(
    outputs: QuillonOutputs, custody_root: Path
) -> None:
    result = _write_checkpoint(custody_root)

    entries = _entries(outputs, "warm-cohort", [_reference(result, custody_root)])

    assert len(entries) == 1
    entry = entries[0]
    assert entry.node_key == "training:warm-cohort"
    assert entry.row_id == ROW
    assert entry.role_path == ("rows", ROW, "checkpoint_initialization")
    assert entry.mode == "continue_from"
    assert entry.source.kind == "TrainingCheckpointTransactionManifest"
    assert entry.source.id == result.manifest.transaction_id
    assert entry.source.role == CHECKPOINT_CUSTODY_ROLE
    assert entry.source.uri is None
    assert entry.source.metadata["manifest_sha256"] == sha256_bytes(
        result.manifest_path.read_bytes()
    )
    assert entry.record()["mode"] == "continue_from"


def test_the_read_only_entrypoint_needs_no_receipt_root(
    outputs: QuillonOutputs, custody_root: Path
) -> None:
    result = _write_checkpoint(custody_root)
    outputs.cohort("entrypoint-cohort", references=[_reference(result, custody_root)])

    entries = experiment_checkpoint_initializations(
        "entrypoint-cohort", output_directory=outputs.output_directory
    )

    assert [entry.row_id for entry in entries] == [ROW]
    assert entries[0].source.id == result.manifest.transaction_id


def test_two_rows_each_carry_their_own_mode(
    outputs: QuillonOutputs, custody_root: Path
) -> None:
    result = _write_checkpoint(custody_root)

    entries = _entries(
        outputs,
        "mixed-cohort",
        [
            _reference(result, custody_root, mode="continue_from", row="row-a"),
            _reference(result, custody_root, mode="initialize_from", row="row-b"),
        ],
    )

    assert [(entry.row_id, entry.mode) for entry in entries] == [
        ("row-a", "continue_from"),
        ("row-b", "initialize_from"),
    ]


# --------------------------------------------------------------------------
# Execution: the source is authenticated, then lowered
# --------------------------------------------------------------------------


def test_the_source_authenticates_and_lowers_against_its_own_structure(
    outputs: QuillonOutputs, custody_root: Path
) -> None:
    result = _write_checkpoint(custody_root)
    entry = _entries(outputs, "lowered-cohort", [_reference(result, custody_root)])[0]
    context = _context(custody_root)

    authenticated = authenticate_checkpoint_initialization_source(
        entry, execution_context=context, custody_binding=CUSTODY_BINDING
    )
    assert authenticated.manifest.transaction_id == result.manifest.transaction_id

    structure = checkpoint_initialization_source_structure(
        entry, execution_context=context, custody_binding=CUSTODY_BINDING
    )
    plan = lower_planned_checkpoint_initialization(
        entry,
        target=structure,
        execution_context=context,
        custody_binding=CUSTODY_BINDING,
    )

    assert plan.mode == "continue_from"
    assert plan.source == entry.source
    assert plan.fresh_paths == ()
    assert set(plan.restored_paths) == {
        f"{slot.slot}/{leaf.path}"
        for slot in structure.slots
        for leaf in slot.leaves
    }


def test_a_warm_start_restores_only_the_model_slot(
    outputs: QuillonOutputs, custody_root: Path
) -> None:
    result = _write_checkpoint(custody_root)
    entry = _entries(
        outputs, "restart-cohort", [_reference(result, custody_root, mode="initialize_from")]
    )[0]
    context = _context(custody_root)
    structure = checkpoint_initialization_source_structure(
        entry, execution_context=context, custody_binding=CUSTODY_BINDING
    )

    plan = lower_planned_checkpoint_initialization(
        entry,
        target=structure,
        execution_context=context,
        custody_binding=CUSTODY_BINDING,
    )

    restored_slots = {entry.slot for entry in plan.slots if entry.action == "restore"}
    model_slots = {slot.slot for slot in structure.slots if slot.role == "model"}
    assert restored_slots == model_slots
    assert model_slots


# --------------------------------------------------------------------------
# Fail closed: absent, altered, or unauthenticatable
# --------------------------------------------------------------------------


def test_an_absent_checkpoint_refuses(
    outputs: QuillonOutputs, custody_root: Path, tmp_path: Path
) -> None:
    result = _write_checkpoint(custody_root)
    entry = _entries(outputs, "absent-cohort", [_reference(result, custody_root)])[0]
    empty = tmp_path / "empty-custody"
    empty.mkdir()

    with pytest.raises(CheckpointInitializationWiringError, match="does not authenticate it"):
        authenticate_checkpoint_initialization_source(
            entry, execution_context=_context(empty), custody_binding=CUSTODY_BINDING
        )


def test_an_altered_checkpoint_manifest_refuses(
    outputs: QuillonOutputs, custody_root: Path
) -> None:
    result = _write_checkpoint(custody_root)
    entry = _entries(outputs, "altered-cohort", [_reference(result, custody_root)])[0]
    payload = json.loads(result.manifest_path.read_text(encoding="utf-8"))
    payload["metadata"] = {**payload.get("metadata", {}), "tampered": True}
    result.manifest_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    with pytest.raises(CheckpointInitializationWiringError, match="does not authenticate it"):
        authenticate_checkpoint_initialization_source(
            entry,
            execution_context=_context(custody_root),
            custody_binding=CUSTODY_BINDING,
        )


def test_an_unbound_custody_binding_refuses(
    outputs: QuillonOutputs, custody_root: Path
) -> None:
    result = _write_checkpoint(custody_root)
    entry = _entries(outputs, "unbound-cohort", [_reference(result, custody_root)])[0]

    with pytest.raises(StagedExecutionContextError, match="unavailable"):
        authenticate_checkpoint_initialization_source(
            entry,
            execution_context=_context(custody_root),
            custody_binding="other-checkpoints",
        )


def test_a_locator_without_a_digest_cannot_authenticate_a_source(
    outputs: QuillonOutputs,
) -> None:
    locator = ReceiptLocatorReference(
        manifest_kind="TrainingCheckpointTransactionManifest",
        manifest_id="feedbax-checkpoint:pending",
        role_path=f"rows.{ROW}.checkpoint_initialization",
        consumer=CheckpointInitializationBinding(mode="continue_from", row_id=ROW),
    )
    with pytest.raises(CheckpointInitializationWiringError, match="quotes no byte profile"):
        _entries(outputs, "pending-cohort", [locator])


def test_a_source_without_an_execution_location_refuses(
    outputs: QuillonOutputs, custody_root: Path
) -> None:
    result = _write_checkpoint(custody_root)
    with pytest.raises(CheckpointInitializationWiringError, match="execution_uri"):
        _entries(
            outputs,
            "unlocated-cohort",
            [_reference(result, custody_root, execution_uri=None)],
        )


def test_a_source_that_is_not_a_checkpoint_refuses(
    outputs: QuillonOutputs, custody_root: Path
) -> None:
    result = _write_checkpoint(custody_root)
    with pytest.raises(CheckpointInitializationWiringError, match="TrainingCheckpointTransaction"):
        _entries(
            outputs,
            "wrong-kind-cohort",
            [_reference(result, custody_root, manifest_kind="TrainingRunManifest")],
        )


def test_two_references_for_one_row_refuse(
    outputs: QuillonOutputs, custody_root: Path
) -> None:
    result = _write_checkpoint(custody_root)
    first = _reference(result, custody_root)
    second = _reference(result, custody_root).model_copy(
        update={"role_path": f"rows.{ROW}.checkpoint_initialization.again"}
    )
    with pytest.raises(CheckpointInitializationWiringError, match="exactly one checkpoint"):
        _entries(outputs, "double-cohort", [first, second])


def test_a_checkpoint_initialization_may_not_be_declared_off_the_training_layer(
    outputs: QuillonOutputs, custody_root: Path
) -> None:
    """A checkpoint initializes a training row, so no other layer declares one."""
    result = _write_checkpoint(custody_root)
    outputs.probe("misplaced", references=[_reference(result, custody_root)])
    index = read_compiled_outputs(outputs.output_directory)

    with pytest.raises(CheckpointInitializationWiringError, match="layer product"):
        compiled_checkpoint_initializations(index.resolve_target("misplaced"))
