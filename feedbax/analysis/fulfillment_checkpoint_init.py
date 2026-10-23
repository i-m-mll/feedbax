"""Wiring an authored checkpoint initialization into the request that runs it.

An envelope may state that one training row starts from an existing checkpoint —
``initialize_from`` for a model-weight warm start, ``continue_from`` for a
continuation. The compile records that statement the same way it records every
other cross-layer fact: as one typed reference on the training layer's compile
lock, carrying a
:class:`~feedbax.contracts.experiment_compile_lock.CheckpointInitializationBinding`
that names the row and the mode. This module is the only thing that turns those
references into
:class:`~feedbax.contracts.checkpoint_initialization.CheckpointInitializationRequest`
objects, and the only thing that authenticates the checkpoints they name.

## Why this is not a local artifact operation

Training is an external workflow operation: :mod:`feedbax.workflow.execution`
does not launch it, so a plan that names one is refused before anything runs.
A checkpoint initialization is therefore never lowered into a local operation —
:func:`~feedbax.workflow.operation_execution.binding_role` refuses exactly that
— and is instead read off the plan here, by the orchestration entrypoint that
launches the rows. Reaching a checkpoint initialization from any layer other
than training is refused for the same reason: a checkpoint initializes a
training row and nothing else.

## What authenticates the source

:class:`~feedbax.contracts.checkpoint_initialization.CheckpointInitializationRequest`
demands an authenticated source, and only a run that already wrote the checkpoint
can supply one. So the reference this reads must be an
:class:`~feedbax.contracts.experiment_compile_lock.AuthenticatedReceiptReference`:
a locator with no digest states a checkpoint nobody has proved exists, and a
planned product states one that has not been written yet. Both refuse.

The request's ``source`` is the portable, locator-free authenticated ref. Where
those bytes actually live is a separate, machine-local fact: the reference's
``execution_uri``, resolved against a checkpoint-custody root the caller's staged
execution context explicitly binds. Authentication reads the manifest bytes and
every slot blob under that root, so an absent, moved, truncated, or altered
checkpoint refuses here rather than surfacing as a surprise mid-launch.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

from feedbax.workflow.derivation import (
    CompiledEnvelope,
    CompiledOutputIndex,
    WorkflowDerivationError,
)
from feedbax.workflow.experiment import plan_experiment_workflow
from feedbax.contracts.checkpoint_initialization import (
    CheckpointInitializationPlan,
    CheckpointInitializationRequest,
    CheckpointStructure,
    checkpoint_structure_from_manifest,
    lower_checkpoint_initialization,
)
from feedbax.contracts.experiment_compile_lock import (
    AuthenticatedReceiptReference,
    CheckpointInitializationBinding,
)
from feedbax.contracts.base import (
    AUTHENTICATED_MANIFEST_REF_SCHEMA_ID,
    AUTHENTICATED_MANIFEST_REF_SCHEMA_VERSION,
    ParentRef,
)
from feedbax.contracts.staged_execution import validate_staged_binding_name
from feedbax.training.checkpoint_custody import (
    AuthenticatedCheckpointTransaction,
    CheckpointReferenceResolutionError,
    authenticate_checkpoint_custody_ref,
)

if TYPE_CHECKING:  # pragma: no cover - typing only
    from feedbax.analysis.execution_context import StagedExecutionContext
    from feedbax.workflow.plan import WorkflowPlan

#: The manifest kind a checkpoint initialization source names. A checkpoint is
#: held by its transaction manifest, not by the run manifest of the job that
#: wrote it, so this is the one kind the source may be.
CHECKPOINT_SOURCE_MANIFEST_KIND = "TrainingCheckpointTransactionManifest"

#: The role checkpoint custody addresses its trusted references by. It is
#: custody's own vocabulary, restated here so both refs this module mints carry
#: the one role custody accepts.
CHECKPOINT_CUSTODY_ROLE = "training_checkpoint_custody"

#: The layer a checkpoint initialization may be declared on.
CHECKPOINT_INITIALIZATION_LAYER = "campaign"


class CheckpointInitializationWiringError(WorkflowDerivationError):
    """An authored checkpoint initialization cannot be wired into a request."""


@dataclass(frozen=True)
class PlannedCheckpointInitialization:
    """One row's checkpoint initialization, as a request plus where to prove it.

    Attributes:
        node_key: The logical key of the training node that declares it.
        row_id: The row the checkpoint initializes.
        role_path: The lock role path the reference was stated at.
        request: The portable request, whose source is authenticated and carries
            no machine-local locator.
        execution_uri: The custody-root-relative location of the source
            transaction manifest, as the compile quoted it.
    """

    node_key: str
    row_id: str
    role_path: tuple[str, ...]
    request: CheckpointInitializationRequest
    execution_uri: str

    @property
    def mode(self) -> str:
        """Whether the row warm-starts from or continues the source checkpoint."""
        return self.request.mode

    @property
    def source(self) -> ParentRef:
        """The authenticated, locator-free reference the request initializes from."""
        return self.request.source

    def custody_ref(self, *, custody_binding: str) -> ParentRef:
        """Return the locator-bearing reference custody authenticates this by.

        The portable request keeps machine-local locators out of its source, and
        checkpoint custody needs one. This is the same identity and the same byte
        profile, addressed at the location the compile quoted, under the custody
        binding the caller declares.
        """
        validate_staged_binding_name(custody_binding)
        return ParentRef(
            kind=self.request.source.kind,
            id=self.request.source.id,
            role=CHECKPOINT_CUSTODY_ROLE,
            uri=self.execution_uri,
            metadata={
                "checkpoint_custody_binding": custody_binding,
                "manifest_sha256": self.request.source.metadata["manifest_sha256"],
            },
        )

    def record(self) -> dict[str, Any]:
        """Return the deterministic structured record of this initialization."""
        return {
            "node_key": self.node_key,
            "row_id": self.row_id,
            "role_path": list(self.role_path),
            "mode": self.request.mode,
            "source": self.request.source.model_dump(mode="json", exclude_none=True),
            "execution_uri": self.execution_uri,
        }


def compiled_checkpoint_initializations(
    compiled: CompiledEnvelope, *, node_key: str | None = None
) -> tuple[PlannedCheckpointInitialization, ...]:
    """Return every checkpoint initialization one compiled output's lock states.

    Order is the lock's own reference order, so two reads of one compile produce
    the same entries in the same order.

    Raises:
        CheckpointInitializationWiringError: The binding appears on a non-training
            product, its reference cannot authenticate a source, it names
            something other than a checkpoint transaction manifest, it quotes no
            execution location, or two references initialize one row.
    """
    ref = str(compiled.lock_path)
    key = compiled.key.text if node_key is None else node_key
    entries: list[PlannedCheckpointInitialization] = []
    rows: dict[str, str] = {}
    for reference in compiled.plan_edge_references():
        consumer = getattr(reference, "consumer", None)
        if not isinstance(consumer, CheckpointInitializationBinding):
            continue
        role_path = str(reference.role_path)
        if compiled.kind.layer != CHECKPOINT_INITIALIZATION_LAYER:
            raise CheckpointInitializationWiringError(
                f"{ref} initializes row {consumer.row_id!r} from a checkpoint on a "
                f"{compiled.kind.layer!r} layer product. A checkpoint initializes a campaign "
                "row, so only a campaign layer product declares one."
            )
        if not isinstance(reference, AuthenticatedReceiptReference):
            raise CheckpointInitializationWiringError(
                f"{ref} initializes row {consumer.row_id!r} at role {role_path!r} from a "
                f"{type(reference).__name__}, which quotes no byte profile. A checkpoint "
                "initialization demands an authenticated source, and only a run that already "
                "wrote the checkpoint supplies one."
            )
        if reference.manifest_kind != CHECKPOINT_SOURCE_MANIFEST_KIND:
            raise CheckpointInitializationWiringError(
                f"{ref} initializes row {consumer.row_id!r} from a "
                f"{reference.manifest_kind!r}; a checkpoint is held by its "
                f"{CHECKPOINT_SOURCE_MANIFEST_KIND!r}, which is what the source names"
            )
        if reference.execution_uri is None:
            raise CheckpointInitializationWiringError(
                f"{ref} initializes row {consumer.row_id!r} from checkpoint "
                f"{reference.manifest_id!r} without quoting where it was executed from. A "
                "checkpoint lives in a custody tree addressed root-relative, so the reference "
                "states its execution_uri or nothing can authenticate it."
            )
        previous = rows.get(consumer.row_id)
        if previous is not None:
            raise CheckpointInitializationWiringError(
                f"{ref} initializes row {consumer.row_id!r} at both {previous!r} and "
                f"{role_path!r}; a row starts from exactly one checkpoint"
            )
        rows[consumer.row_id] = role_path
        entries.append(
            PlannedCheckpointInitialization(
                node_key=key,
                row_id=consumer.row_id,
                role_path=tuple(role_path.split(".")),
                request=CheckpointInitializationRequest(
                    mode=consumer.mode,
                    source=ParentRef(
                        kind=reference.manifest_kind,
                        id=reference.manifest_id,
                        role=CHECKPOINT_CUSTODY_ROLE,
                        metadata={
                            "ref_schema_id": AUTHENTICATED_MANIFEST_REF_SCHEMA_ID,
                            "ref_schema_version": AUTHENTICATED_MANIFEST_REF_SCHEMA_VERSION,
                            "manifest_sha256": reference.manifest_sha256,
                            "size_bytes": reference.size_bytes,
                        },
                    ),
                ),
                execution_uri=reference.execution_uri,
            )
        )
    return tuple(entries)


def checkpoint_initialization_requests(
    plan: "WorkflowPlan", *, index: CompiledOutputIndex
) -> tuple[PlannedCheckpointInitialization, ...]:
    """Return every checkpoint initialization one plan's closure states, in order.

    The plan is read, not executed: training is an external operation that the
    local workflow executor never runs, and this is the surface its invocation
    entrypoint reads its warm starts and continuations from.
    """
    entries: list[PlannedCheckpointInitialization] = []
    for node in plan.nodes:
        entries.extend(
            compiled_checkpoint_initializations(
                index.require(node.source_ref), node_key=node.key.text
            )
        )
    return tuple(entries)


def experiment_checkpoint_initializations(
    target: str, *, output_directory: Path | str
) -> tuple[PlannedCheckpointInitialization, ...]:
    """Read one compiled target's checkpoint initializations from its outputs.

    This is the read-only counterpart of
    :func:`~feedbax.workflow.experiment.plan_experiment_workflow` for
    a launch entrypoint: nothing executes, nothing is written, and no receipt
    root is needed to learn which rows start from which checkpoints.
    """
    plan, index = plan_experiment_workflow(target, output_directory=output_directory)
    return checkpoint_initialization_requests(plan, index=index)


def authenticate_checkpoint_initialization_source(
    entry: PlannedCheckpointInitialization,
    *,
    execution_context: "StagedExecutionContext",
    custody_binding: str,
) -> AuthenticatedCheckpointTransaction:
    """Authenticate one initialization's source under an explicitly bound root.

    The custody root is the caller's declaration, never synthesized here. Every
    slot blob the transaction declares is verified along with the manifest bytes,
    and nothing is deserialized.

    Raises:
        CheckpointInitializationWiringError: The checkpoint is absent from the
            bound root, or its bytes do not match what the compile quoted.
    """
    root = execution_context.checkpoint_custody_root(custody_binding)
    try:
        return authenticate_checkpoint_custody_ref(
            entry.custody_ref(custody_binding=custody_binding), allowed_root=root
        )
    except CheckpointReferenceResolutionError as exc:
        raise CheckpointInitializationWiringError(
            f"row {entry.row_id!r} initializes from checkpoint {entry.source.id!r} at "
            f"{entry.execution_uri!r} under custody binding {custody_binding!r}, which does "
            f"not authenticate it: {exc}. A checkpoint that cannot be authenticated is "
            "absent, moved, or altered; it never means the row starts fresh."
        ) from exc


def checkpoint_initialization_source_structure(
    entry: PlannedCheckpointInitialization,
    *,
    execution_context: "StagedExecutionContext",
    custody_binding: str,
) -> CheckpointStructure:
    """Return the canonical slot structure one authenticated source checkpoint has."""
    authenticated = authenticate_checkpoint_initialization_source(
        entry, execution_context=execution_context, custody_binding=custody_binding
    )
    return checkpoint_structure_from_manifest(authenticated.manifest)


def lower_planned_checkpoint_initialization(
    entry: PlannedCheckpointInitialization,
    *,
    target: CheckpointStructure,
    execution_context: "StagedExecutionContext",
    custody_binding: str,
) -> CheckpointInitializationPlan:
    """Lower one planned initialization against the structure the row will build.

    The closed matching rule is
    :func:`~feedbax.contracts.checkpoint_initialization.lower_checkpoint_initialization`'s;
    this adds only the authenticated source it needs.
    """
    source = checkpoint_initialization_source_structure(
        entry, execution_context=execution_context, custody_binding=custody_binding
    )
    return lower_checkpoint_initialization(entry.request, source=source, target=target)


def checkpoint_initializations_by_row(
    entries: Sequence[PlannedCheckpointInitialization],
) -> dict[str, PlannedCheckpointInitialization]:
    """Index planned initializations by the row each one starts, refusing collisions."""
    indexed: dict[str, PlannedCheckpointInitialization] = {}
    for entry in entries:
        previous = indexed.get(entry.row_id)
        if previous is not None:
            raise CheckpointInitializationWiringError(
                f"row {entry.row_id!r} is initialized by both {previous.node_key} and "
                f"{entry.node_key}; a row starts from exactly one checkpoint"
            )
        indexed[entry.row_id] = entry
    return indexed


__all__ = [
    "CHECKPOINT_CUSTODY_ROLE",
    "CHECKPOINT_INITIALIZATION_LAYER",
    "CHECKPOINT_SOURCE_MANIFEST_KIND",
    "CheckpointInitializationWiringError",
    "PlannedCheckpointInitialization",
    "authenticate_checkpoint_initialization_source",
    "checkpoint_initialization_requests",
    "checkpoint_initialization_source_structure",
    "checkpoint_initializations_by_row",
    "compiled_checkpoint_initializations",
    "experiment_checkpoint_initializations",
    "lower_planned_checkpoint_initialization",
]
