"""Prepare a declared training invocation for the CLI or job worker."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import TYPE_CHECKING, Any, Literal

from feedbax.compiler import DocumentRoot, GraphDocument, compile_graph
from feedbax.compiler.prototypes import prototypes_from_task_bindings
from feedbax.contracts.authored_canonical import canonical_sha256
from feedbax.contracts.graph import GraphSpec
from feedbax.contracts.studio_training import STUDIO_GRAPH_METHOD, StudioGraphMethodPayload
from feedbax.contracts.training import TrainingRunSpec, resolve_training_run_spec
from feedbax.training.preparation import (
    ExecutionPreparationPlan,
    ExecutionPreparationRequest,
    ExecutionPreparationResult,
    MaterializedExecutionPreparation,
    lower_zero_level_preparation_plan,
    materialize_execution_preparation,
    require_execution_preparation_provider,
)
from feedbax.training.studio_executor import studio_training_key
from feedbax.training.worker_validation import resolve_execution_mapping
from feedbax.persistence.model_assets import model_artifact_provider

if TYPE_CHECKING:
    from feedbax.contracts.artifact_custody import ArtifactBlobProvider
    from feedbax.plugins.application import ApplicationRegistryBundle


def prepare_training_invocation(
    run_spec: TrainingRunSpec,
    registries: ApplicationRegistryBundle,
    *,
    run_id: str | None,
    resume: bool,
    initial_slots: Mapping[str, Any] | None = None,
    model_assets: ArtifactBlobProvider | None = None,
    phase: Callable[[Literal["building", "compiling"]], None] | None = None,
) -> ExecutionPreparationResult | MaterializedExecutionPreparation | None:
    if run_spec.trial_documents is not None:
        raise ValueError("/trial_documents: The retained envelope cannot prepare managed requests.")
    resolved_method = resolve_training_run_spec(run_spec, registries.training_programs)
    if run_spec.graph.inline is None:
        raise ValueError("execute-training-run-spec requires an inline graph document")
    registration = registries.execution_preparations.get(run_spec.method_ref.key)
    mapping_levels, _ = resolve_execution_mapping(run_spec.worker_execution)
    if resolved_method.registration.requires_execution_preparation:
        require_execution_preparation_provider(
            method_ref=run_spec.method_ref.key,
            preparation_registry=registries.execution_preparations,
        )
    if initial_slots is not None:
        if mapping_levels:
            raise ValueError(
                "--initial-slots is invalid when mapping_levels are active; mapped runs require "
                "a Feedbax-owned execution-preparation plan"
            )
        if registration is not None:
            raise ValueError(
                "--initial-slots cannot be combined with an execution-preparation provider for "
                f"method_ref {run_spec.method_ref.key!r}"
            )

    if phase is not None:
        phase("building")
    graph_document = GraphDocument(
        graph=run_spec.graph.inline,
        trial_root=DocumentRoot(
            schema_id="feedbax.spec.training.task",
            schema_version="1",
            content_sha256=canonical_sha256(run_spec.task.model_dump(mode="json")),
        ),
        objective_root=DocumentRoot(
            schema_id=run_spec.objective.schema_id or "feedbax.objective_slot",
            schema_version=run_spec.objective.schema_version or "1",
            content_sha256=canonical_sha256(
                run_spec.objective.model_dump(mode="json", exclude_none=True)
            ),
        ),
    )
    studio_method = run_spec.method_ref == STUDIO_GRAPH_METHOD
    payload = resolved_method.payload
    executable_graph = compile_graph(
        graph_document,
        registries.components,
        input_prototypes=(
            prototypes_from_task_bindings(run_spec.task_binding_spec)
            if run_spec.task_binding_spec is not None
            else None
        ),
        initialization_key=(
            studio_training_key(payload.seed, "model")
            if studio_method and isinstance(payload, StudioGraphMethodPayload)
            else None
        ),
        artifact_provider=model_assets or (
            model_artifact_provider(GraphSpec.model_validate(graph_document.graph))
            if studio_method
            else None
        ),
    )
    if registration is None:
        return None

    request = ExecutionPreparationRequest(
        run_spec=run_spec,
        executable_graph=executable_graph,
        method_payload=payload,
        method_contract=resolved_method.contract,
        effective_phase=resolved_method.effective_phase,
        run_id=run_id,
        resume=resume,
    )
    if phase is not None:
        phase("compiling")
    prepared = registries.execution_preparations.prepare(request)
    if isinstance(prepared, ExecutionPreparationPlan):
        if mapping_levels:
            return materialize_execution_preparation(
                request, prepared, provider_identity=registration.owner
            )
        return lower_zero_level_preparation_plan(prepared, resume_template=resume)
    return prepared
