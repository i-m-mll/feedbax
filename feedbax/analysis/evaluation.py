"""Registered execution for manifest-canonical evaluation runs.

Producer contract:

1. Evaluation producers register lowercase dotted ``evaluation_type`` keys of
   the form ``<package>.<name>``. Feedbax-owned recipes use ``feedbax.*`` keys;
   downstream packages use their import-name prefix, for example ``rlrmp.*``.
2. A conforming producer registers a params schema family for
   ``<package>.spec.evaluation.<name>``. Feedbax v1 conformance allows an
   explicit waiver for un-schema'd params; strict executor rejection is a later
   ratchet.
3. Recipes that return states declare
   ``EvaluationRecipeResult.metadata["states_schema"]`` as an opaque non-empty
   string identifying the states pytree shape.
4. Recipes are deterministic for the same ``EvaluationRunSpec``, avoid durable
   writes inside the recipe, do not depend on ambient CWD or global config, and
   access model/artifact inputs only through resolved ``ParentRef`` inputs.
"""

from __future__ import annotations

import copy
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field, replace
import json
from math import prod
from pathlib import Path
import shutil
import tempfile
from typing import Any, Sequence

import dill as pickle
from pydantic import BaseModel, Field, model_validator

from feedbax.contracts.evaluation_states import (
    EVALUATION_STATES_ARTIFACT_ROLE,
    load_evaluation_states_artifact,
    store_evaluation_states_artifact,
)
from feedbax.contracts.manifest import (
    ArtifactRef,
    EntrypointRef,
    EvaluationRunManifest,
    EvaluationRunSpec,
    StagedEvaluationPrerequisite,
    ManifestStatus,
    Provenance,
    canonical_manifest_path,
    collect_git_provenance,
    default_manifest_root,
    evaluation_run_manifest_id,
    evaluation_states_cache_path,
    load_manifest,
    spec_payload,
    write_manifest,
    EVALUATION_RUN_MATRIX_SPEC_SCHEMA_ID,
    EVALUATION_RUN_MATRIX_SPEC_SCHEMA_VERSION,
    EVALUATION_AXIS_EXPANSION_PROVENANCE_SCHEMA_ID,
    EVALUATION_AXIS_EXPANSION_PROVENANCE_SCHEMA_VERSION,
    StrictModel,
    canonical_json_bytes,
    sha256_bytes,
)
from feedbax.contracts.evaluation_composition import (
    EvaluationRunMatrixDeltaSpec,
    FlattenedEvaluationMatrix,
    evaluation_matrix_composition_provenance,
    flatten_evaluation_run_matrix_delta,
    is_evaluation_run_matrix_delta_payload,
)
from feedbax.contracts.extraction import SourceBinding
from feedbax.contracts.matrix_core import (
    ContentPinnedJsonBase,
    MaterializedMatrixRow,
    MatrixAxis,
    MatrixRow,
    RowDerivation,
    RowMatrixSpec,
    expand_matrix_axes,
    load_content_pinned_json_base,
    materialize_matrix_rows,
)
from feedbax.contracts.schema_namespace import (
    validate_schema_identity,
    validate_schema_version,
)
from feedbax.analysis.declared_versions import migrate_authored_document
from feedbax.analysis.execution_context import (
    EMPTY_STAGED_EXECUTION_CONTEXT,
    StagedArtifactProviderRootBinding,
    StagedCheckpointCustodyRootBinding,
    StagedExecutionContext,
    StagedExecutionContextError,
    StagedParentExecutionLocation,
    resolve_staged_execution_context,
    with_staged_parent_execution_locations,
    with_staged_repo_root,
)
from feedbax.analysis.manifest_inputs import (
    is_authenticated_manifest_ref,
    restated_parent_differences,
    resolve_manifest_input,
)
from feedbax.analysis.evaluation_inputs import resolve_evaluation_inputs
from feedbax.contracts.staged_execution import (
    StagedExecutionDescriptor,
    validate_staged_binding_name,
)
from feedbax.analysis.validation import (
    EvaluationRecipeProtocol,
    validate_evaluation_batch_recipe,
    validate_evaluation_recipe,
    validate_namespaced_type_key,
)


@dataclass(frozen=True)
class EvaluationRecipeResult:
    """Result returned by an evaluation recipe.

    ``metadata["states_schema"]`` is a reserved producer-declared identifier
    for the returned states pytree shape. Runtime execution treats it as opaque;
    the conformance suite requires it when a recipe returns states.
    """

    states: Any = None
    summary_metrics: dict[str, Any] = field(default_factory=dict)
    artifacts: list[ArtifactRef] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


EvaluationRecipe = EvaluationRecipeProtocol
EvaluationBatchRecipe = Callable[
    [Sequence["EvaluationBatchItem"], StagedExecutionContext], Sequence[EvaluationRecipeResult]
]
STATES_SCHEMA_METADATA_KEY = "states_schema"
EVALUATION_STATES_CACHE_SCHEMA_VERSION = "feedbax.analysis.evaluation-states-cache.v1"
_MAX_FAILURE_DIAGNOSTICS_BYTES = 16 * 1024


class EvaluationRecipeRegistry:
    """Isolated evaluation, batch, and authoring-schema registry."""

    def __init__(self) -> None:
        self._sealed = False
        self._recipes: dict[str, EvaluationRecipe] = {}
        self._batch_recipes: dict[str, EvaluationBatchRecipe] = {}
        self._schemas: dict[str, EvaluationAuthoringSchema] = {}

    def register(
        self,
        evaluation_type: str,
        recipe: EvaluationRecipe,
        *,
        batch_recipe: EvaluationBatchRecipe | None = None,
    ) -> None:
        if self._sealed:
            raise RuntimeError("evaluation recipe registry is sealed")
        evaluation_type = validate_namespaced_type_key(evaluation_type, field="evaluation_type")
        if evaluation_type in self._recipes:
            raise ValueError(f"Evaluation recipe {evaluation_type!r} is already registered")
        self._recipes[evaluation_type] = validate_evaluation_recipe(evaluation_type, recipe)
        if batch_recipe is not None:
            self._batch_recipes[evaluation_type] = validate_evaluation_batch_recipe(
                evaluation_type, batch_recipe
            )

    def register_authoring_schema(
        self, evaluation_type: str, schema: EvaluationAuthoringSchema
    ) -> None:
        if self._sealed:
            raise RuntimeError("evaluation recipe registry is sealed")
        evaluation_type = validate_namespaced_type_key(evaluation_type, field="evaluation_type")
        if not isinstance(schema, EvaluationAuthoringSchema):
            raise TypeError("schema must be an EvaluationAuthoringSchema")
        if evaluation_type in self._schemas:
            raise ValueError(
                f"Evaluation authoring schema {evaluation_type!r} is already registered"
            )
        self._schemas[evaluation_type] = copy.deepcopy(schema)

    def keys(self) -> tuple[str, ...]:
        return tuple(sorted(self._recipes))

    def get(self, evaluation_type: str) -> EvaluationRecipe:
        try:
            return self._recipes[evaluation_type]
        except KeyError as exc:
            raise ValueError(
                f"Evaluation recipe {evaluation_type!r} is not registered; "
                f"available={list(self.keys())!r}"
            ) from exc

    def batch(self, evaluation_type: str) -> EvaluationBatchRecipe:
        try:
            return self._batch_recipes[evaluation_type]
        except KeyError as exc:
            raise ValueError(
                f"Evaluation recipe {evaluation_type!r} has no registered batch recipe"
            ) from exc

    def authoring_schema(self, evaluation_type: str) -> EvaluationAuthoringSchema | None:
        schema = self._schemas.get(evaluation_type)
        return copy.deepcopy(schema) if schema is not None else None

    def seal(self) -> None:
        self._sealed = True


@dataclass(frozen=True)
class EvaluationBatchExecution:
    """Opt into process-atomic publication, not fsync or power-loss durability.

    Hard kills may leave sibling ``.{root}.batch-*`` directories; delete one only
    after confirming no live batch owns it. Batch mode requires a fresh root;
    grids needing reuse or resume must use per-row mode.
    """

    ordered_row_ids: tuple[str, ...] | None = None
    batch_units: tuple[tuple[str, tuple[str, ...]], ...] | None = None


@dataclass(frozen=True)
class EvaluationBatchItem:
    """One addressable row supplied to a registered batch recipe."""

    row_id: str
    spec: EvaluationRunSpec


class EvaluationBatchRowError(RuntimeError):
    """Fail-closed batch diagnostic identifying the row that failed."""

    def __init__(self, row_id: str, cause: BaseException):
        super().__init__(row_id, cause)
        self.row_id = row_id
        self.cause = cause
        self.__cause__ = cause

    def __str__(self) -> str:
        """Render the row-scoped cause while keeping pickle reconstruction arguments."""
        return f"evaluation batch row {self.row_id!r} failed: {self.cause}"


@dataclass(frozen=True)
class EvaluationAuthoringSchema:
    """Runtime-only downstream schema enforced while evaluation matrices compile."""

    schema_id: str
    schema_version: str
    params_model: type[BaseModel]
    axis_profiles: tuple[Mapping[str, tuple[str, ...]], ...]

    def __post_init__(self) -> None:
        family = "evaluation authoring schema"
        validate_schema_identity(self.schema_id, family=family)
        validate_schema_version(self.schema_version, family=family)
        if not self.schema_version.startswith(f"{self.schema_id}.v"):
            raise ValueError("authoring schema_version must be versioned under schema_id")
        if not isinstance(self.params_model, type) or not issubclass(self.params_model, BaseModel):
            raise TypeError("authoring params_model must be a Pydantic BaseModel type")
        if not self.axis_profiles:
            raise ValueError("authoring schema requires at least one axis profile")
        for profile in self.axis_profiles:
            if not profile or any(
                not isinstance(name, str)
                or not name
                or not values
                or any(not isinstance(value, str) or not value for value in values)
                or len(values) != len(set(values))
                for name, values in profile.items()
            ):
                raise ValueError("authoring axes require non-empty unique string value IDs")


class EvaluationRunMatrixSpec(StrictModel):
    """Durable explicit-row or content-pinned axis authoring contract."""

    schema_id: str = EVALUATION_RUN_MATRIX_SPEC_SCHEMA_ID
    schema_version: str = EVALUATION_RUN_MATRIX_SPEC_SCHEMA_VERSION
    base: EvaluationRunSpec | ContentPinnedJsonBase
    rows: list[MatrixRow] = Field(default_factory=list)
    axes: list[MatrixAxis] = Field(default_factory=list)
    sources: list[SourceBinding] = Field(default_factory=list)
    derivations: list[RowDerivation] = Field(default_factory=list)
    staged_parents: dict[str, StagedEvaluationPrerequisite] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _validate_matrix(self) -> "EvaluationRunMatrixSpec":
        if self.schema_id != EVALUATION_RUN_MATRIX_SPEC_SCHEMA_ID:
            raise ValueError(f"unsupported EvaluationRunMatrixSpec schema_id {self.schema_id!r}")
        if self.schema_version != EVALUATION_RUN_MATRIX_SPEC_SCHEMA_VERSION:
            raise ValueError(
                f"unsupported EvaluationRunMatrixSpec schema_version {self.schema_version!r}"
            )
        if isinstance(self.base, EvaluationRunSpec):
            if not self.rows:
                raise ValueError("explicit evaluation matrix requires rows")
            if self.axes:
                raise ValueError("explicit evaluation matrix cannot also declare axes")
        else:
            if self.rows:
                raise ValueError("axis evaluation matrix cannot also declare explicit rows")
            if not self.axes:
                raise ValueError("content-pinned evaluation matrix requires axes")

        row_ids = [row.row_id for row in self.rows]
        if len(row_ids) != len(set(row_ids)):
            raise ValueError("rows row_id values must be unique")
        axis_ids = [axis.id for axis in self.axes]
        if len(axis_ids) != len(set(axis_ids)):
            raise ValueError("axes id values must be unique")
        aliases = [source.alias for source in self.sources]
        if len(aliases) != len(set(aliases)):
            raise ValueError("sources aliases must be unique")
        paths = [derivation.output_path for derivation in self.derivations]
        if len(paths) != len(set(paths)):
            raise ValueError("derivations output_path values must be unique")
        return self

    @model_validator(mode="after")
    def _validate_staged_parents(self) -> "EvaluationRunMatrixSpec":
        serialized_parents: set[str] = set()
        for name, prerequisite in self.staged_parents.items():
            validate_staged_binding_name(name)
            if not is_authenticated_manifest_ref(prerequisite.parent):
                raise ValueError(
                    f"matrix staged parent {name!r} requires an authenticated ParentRef"
                )
            parent_key = prerequisite.parent.model_dump_json(exclude_none=False)
            if parent_key in serialized_parents:
                raise ValueError("matrix staged parents must contain unique complete ParentRefs")
            serialized_parents.add(parent_key)
        return self


def compile_evaluation_run_matrix(
    spec: "EvaluationRunMatrixSpec | EvaluationRunMatrixDeltaSpec | Mapping[str, Any]",
    *,
    registry: EvaluationRecipeRegistry,
    repo_root: Path | str | None = None,
) -> RowMatrixSpec[EvaluationRunSpec]:
    """Compile durable evaluation authoring to the explicit shared row contract."""
    matrix = _coerce_evaluation_run_matrix_spec(spec, repo_root=repo_root)
    if isinstance(matrix.base, EvaluationRunSpec):
        base = matrix.base
        rows = matrix.rows
    else:
        base = EvaluationRunSpec.model_validate(
            load_content_pinned_json_base(matrix.base, repo_root=repo_root)
        )
        rows = [
            MatrixRow(
                row_id=coordinate.row_id,
                deltas=coordinate.deltas,
                metadata={
                    "axis_product": {
                        "value_indices": coordinate.value_indices,
                        "value_ids": coordinate.value_ids,
                    }
                },
            )
            for coordinate in expand_matrix_axes(matrix.axes)
        ]
    compiled = RowMatrixSpec[EvaluationRunSpec](
        base=base,
        rows=rows,
        sources=matrix.sources,
        derivations=matrix.derivations,
    )
    _validate_evaluation_authoring(matrix, compiled, registry=registry, repo_root=repo_root)
    return compiled


def _validate_evaluation_authoring(
    matrix: EvaluationRunMatrixSpec,
    compiled: RowMatrixSpec[EvaluationRunSpec],
    *,
    registry: EvaluationRecipeRegistry,
    repo_root: Path | str | None,
) -> None:
    evaluation_type = compiled.base.evaluation_type
    schema = registry.authoring_schema(evaluation_type)
    if schema is None:
        return
    actual_axes = {
        axis.id: tuple(value.id for value in axis.resolved_values()) for axis in matrix.axes
    }
    profile = next(
        (profile for profile in schema.axis_profiles if actual_axes == dict(profile)), None
    )
    if profile is None:
        raise ValueError(
            f"evaluation authoring axes for {evaluation_type!r} do not match "
            f"schema {schema.schema_version!r}"
        )
    rows = materialize_matrix_rows(compiled, repo_root=repo_root)
    expected_count = prod(len(values) for values in profile.values())
    if len(rows) != expected_count:
        raise ValueError(
            f"evaluation authoring grid for {evaluation_type!r} is incomplete: "
            f"expected {expected_count} rows, got {len(rows)}"
        )
    for row in rows:
        try:
            schema.params_model.model_validate(copy.deepcopy(row.payload.params))
        except Exception as exc:
            raise ValueError(
                f"evaluation authoring params for row {row.row_id!r} do not match "
                f"schema {schema.schema_version!r}: {exc}"
            ) from exc


def resolve_evaluation_matrix_authoring(
    spec: "EvaluationRunMatrixSpec | EvaluationRunMatrixDeltaSpec | Mapping[str, Any]",
    *,
    repo_root: Path | str | None = None,
) -> tuple[EvaluationRunMatrixSpec, FlattenedEvaluationMatrix | None]:
    """Resolve either evaluation authoring kind to one strict matrix spec.

    Delta-authored documents are flattened against their content-pinned parents
    and then validated as the unchanged ``EvaluationRunMatrixSpec`` contract, so
    every downstream compiler, materializer, and executor path stays identical.

    A direct document is resolved from its own declared ``schema_version``
    through the shared authored-document path, so a supported older matrix
    migrates and an unsupported or malformed declaration fails closed.
    """
    if isinstance(spec, EvaluationRunMatrixSpec):
        return spec, None
    if isinstance(spec, EvaluationRunMatrixDeltaSpec) or is_evaluation_run_matrix_delta_payload(
        spec
    ):
        delta = (
            spec
            if isinstance(spec, EvaluationRunMatrixDeltaSpec)
            else EvaluationRunMatrixDeltaSpec.model_validate(spec)
        )
        flattened = flatten_evaluation_run_matrix_delta(delta, repo_root=repo_root)
        return EvaluationRunMatrixSpec.model_validate(flattened.payload), flattened
    # Matrix documents shipped before the family stamped a version, so a
    # versionless document is admitted as current; every declared version still
    # migrates or fails closed through the shared path.
    migrated = migrate_authored_document(
        "EvaluationRunMatrixSpec",
        spec,
        versionless="accept_as_current",
        path="evaluation_matrix_spec",
    )
    return EvaluationRunMatrixSpec.model_validate(migrated.payload), None


def _coerce_evaluation_run_matrix_spec(
    spec: "EvaluationRunMatrixSpec | EvaluationRunMatrixDeltaSpec | Mapping[str, Any]",
    *,
    repo_root: Path | str | None = None,
) -> EvaluationRunMatrixSpec:
    return resolve_evaluation_matrix_authoring(spec, repo_root=repo_root)[0]


def materialize_evaluation_run_matrix(
    spec: "EvaluationRunMatrixSpec | EvaluationRunMatrixDeltaSpec | Mapping[str, Any]",
    *,
    registry: EvaluationRecipeRegistry,
    repo_root: Path | str | None = None,
) -> list[MaterializedMatrixRow[EvaluationRunSpec]]:
    """Resolve an evaluation matrix into executable evaluation requests."""
    matrix = _coerce_evaluation_run_matrix_spec(spec, repo_root=repo_root)
    compiled = compile_evaluation_run_matrix(matrix, registry=registry, repo_root=repo_root)
    rows = materialize_matrix_rows(compiled, repo_root=repo_root)
    _validate_matrix_staged_parent_references(matrix, rows)
    return rows


def execute_evaluation_run_matrix(
    spec: (
        "EvaluationRunMatrixSpec | EvaluationRunMatrixDeltaSpec | EvaluationRunSpec "
        "| Mapping[str, Any]"
    ),
    *,
    registry: EvaluationRecipeRegistry,
    root: Path | str,
    repo_root: Path | str | None = None,
    escape_hatch_reason: str | None = None,
    parent_manifest_root: Path | str | None = None,
    execution_descriptor: StagedExecutionDescriptor | Mapping[str, Any] | None = None,
    artifact_provider_bindings: Sequence[StagedArtifactProviderRootBinding] = (),
    checkpoint_custody_bindings: Sequence[StagedCheckpointCustodyRootBinding] = (),
    execution_context: StagedExecutionContext | None = None,
    batch: EvaluationBatchExecution | None = None,
    matrix_metadata: Mapping[str, Any] | None = None,
):
    """Resolve and execute every evaluation condition through the shared harness."""
    from feedbax.analysis.harness import MatrixMaterializerHarness

    if execution_context is not None and (
        execution_descriptor is not None
        or artifact_provider_bindings
        or checkpoint_custody_bindings
    ):
        raise StagedExecutionContextError(
            "pre-resolved execution_context cannot be combined with raw staged bindings"
        )
    matrix_metadata = dict(matrix_metadata or {})
    staged_parents: dict[str, Any] = {}
    if isinstance(spec, EvaluationRunSpec) or (
        isinstance(spec, Mapping) and "evaluation_type" in spec and "base" not in spec
    ):
        if escape_hatch_reason is None or not escape_hatch_reason.strip():
            raise ValueError(
                "flat EvaluationRunSpec escape hatch requires a stated non-empty reason"
            )
        run_spec = (
            spec if isinstance(spec, EvaluationRunSpec) else EvaluationRunSpec.model_validate(spec)
        )
        rows = [("flat", run_spec.model_dump(mode="python"))]
        executable_spec = run_spec.model_dump(mode="json", exclude_none=True)
        if execution_context is None:
            execution_context = resolve_staged_execution_context(
                execution_descriptor,
                artifact_provider_bindings=artifact_provider_bindings,
                checkpoint_custody_bindings=checkpoint_custody_bindings,
            )
    else:
        matrix, composition = resolve_evaluation_matrix_authoring(spec, repo_root=repo_root)
        flattened_spec = matrix.model_dump(mode="json", exclude_none=True)
        # Delta-authored runs replay the authored envelope, not its flattened result.
        executable_spec = flattened_spec if composition is None else composition.authored
        compiled = compile_evaluation_run_matrix(matrix, registry=registry, repo_root=repo_root)
        materialized_rows = materialize_matrix_rows(compiled, repo_root=repo_root)
        _validate_matrix_staged_parent_references(matrix, materialized_rows)
        rows = [(row.row_id, row.payload.model_dump(mode="python")) for row in materialized_rows]
        execution_context = _resolve_matrix_staged_parents(
            matrix,
            parent_manifest_root=parent_manifest_root,
            execution_descriptor=execution_descriptor,
            artifact_provider_bindings=artifact_provider_bindings,
            checkpoint_custody_bindings=checkpoint_custody_bindings,
            execution_context=execution_context,
        )
        if isinstance(matrix.base, ContentPinnedJsonBase):
            matrix_metadata["axis_expansion"] = {
                "schema_id": EVALUATION_AXIS_EXPANSION_PROVENANCE_SCHEMA_ID,
                "schema_version": EVALUATION_AXIS_EXPANSION_PROVENANCE_SCHEMA_VERSION,
                "authored_matrix_sha256": sha256_bytes(canonical_json_bytes(flattened_spec)),
                "pinned_base": matrix.base.model_dump(mode="json", exclude_none=True),
                "ordered_axes": [
                    {
                        "axis_id": axis.id,
                        "value_ids": [value.id for value in axis.resolved_values()],
                        # Generator-form axes record the declaration that is the
                        # expansion authority; enumerated axes stay byte-identical.
                        **(
                            {"generator": axis.generator.model_dump(mode="json")}
                            if axis.generator is not None
                            else {}
                        ),
                    }
                    for axis in matrix.axes
                ],
                "coordinates": [
                    {"row_id": row.row_id, **row.metadata["axis_product"]} for row in compiled.rows
                ],
                "canonical_row_order": [row.row_id for row in materialized_rows],
                "canonical_payload_sha256": {
                    row.row_id: sha256_bytes(
                        canonical_json_bytes(row.payload.model_dump(mode="json", exclude_none=True))
                    )
                    for row in materialized_rows
                },
            }
        if composition is not None:
            matrix_metadata["matrix_composition"] = evaluation_matrix_composition_provenance(
                composition,
                flattened_matrix=flattened_spec,
                canonical_row_order=[row.row_id for row in materialized_rows],
                canonical_payload_sha256={
                    row.row_id: sha256_bytes(
                        canonical_json_bytes(row.payload.model_dump(mode="json", exclude_none=True))
                    )
                    for row in materialized_rows
                },
            )
        if matrix.staged_parents:
            matrix_metadata["staged_parents"] = {
                name: prerequisite.model_dump(mode="json", exclude_none=True)
                for name, prerequisite in matrix.staged_parents.items()
            }
        staged_parents = matrix_metadata.get("staged_parents", {})

    execution_context = with_staged_repo_root(execution_context, repo_root)
    regeneration_parameters = {
        "executable_spec": executable_spec,
        "manifest_root": str(Path(root).resolve()),
        "repo_root": str(Path(repo_root).resolve()) if repo_root is not None else None,
        "parent_manifest_root": (
            str(Path(parent_manifest_root)) if parent_manifest_root is not None else None
        ),
        "execution_descriptor": (
            execution_context.descriptor.model_dump(mode="json", exclude_none=True)
            if execution_context.descriptor is not None
            else None
        ),
        "artifact_provider_bindings": [
            {"name": name, "root": str(provider.root)}
            for name, provider in execution_context.opened_artifact_providers.items()
        ],
        "checkpoint_custody_bindings": [
            {"name": name, "root": str(bound_root)}
            for name, bound_root in execution_context.checkpoint_custody_roots.items()
        ],
        "staged_parents": staged_parents,
    }
    if batch is not None:
        if type(batch) is not EvaluationBatchExecution:
            raise TypeError("batch must be an EvaluationBatchExecution instance")
        if escape_hatch_reason is not None:
            raise ValueError("batched execution only accepts EvaluationRunMatrixSpec")
        if batch.ordered_row_ids is not None and batch.batch_units is not None:
            raise ValueError("batch execution accepts either ordered_row_ids or batch_units")
        if batch.batch_units is not None:
            from feedbax.analysis.harness import HarnessResult

            available = {row_id: payload for row_id, payload in rows}
            completed = []
            batch_ids = []
            for batch_id, batch_row_ids in batch.batch_units:
                if not batch_id or batch_id in batch_ids:
                    raise ValueError("batch_units require unique non-empty batch ids")
                unknown = [row_id for row_id in batch_row_ids if row_id not in available]
                if not batch_row_ids or len(batch_row_ids) != len(set(batch_row_ids)) or unknown:
                    raise ValueError("batch_units require non-empty unique known ordered row ids")
                batch_ids.append(batch_id)
                result = _execute_evaluation_batch_rows(
                    [(row_id, available[row_id]) for row_id in batch_row_ids],
                    registry=registry,
                    root=Path(root) / batch_id,
                    execution_context=execution_context,
                    matrix_metadata=matrix_metadata,
                    regeneration_parameters={
                        **regeneration_parameters,
                        "batch_execution": {
                            "mode": "governed_ordered_partition",
                            "batch_id": batch_id,
                            "ordered_row_ids": batch_row_ids,
                        },
                    },
                )
                completed.extend(result.rows)
            return HarnessResult(
                rows=tuple(completed),
                note="Evaluation run matrix persistent-worker partition",
                metadata={"ordered_batch_ids": batch_ids},
            )
        if batch.ordered_row_ids is not None:
            available = {row_id: payload for row_id, payload in rows}
            if len(batch.ordered_row_ids) != len(set(batch.ordered_row_ids)):
                raise ValueError("batch ordered_row_ids must be unique")
            unknown = [row_id for row_id in batch.ordered_row_ids if row_id not in available]
            if unknown:
                raise ValueError(f"batch ordered_row_ids contain unknown matrix rows: {unknown!r}")
            rows = [(row_id, available[row_id]) for row_id in batch.ordered_row_ids]
        return _execute_evaluation_batch_rows(
            rows,
            registry=registry,
            root=Path(root),
            execution_context=execution_context,
            matrix_metadata=matrix_metadata,
            regeneration_parameters={
                **regeneration_parameters,
                "batch_execution": {
                    "mode": (
                        "governed_ordered_subset"
                        if batch.ordered_row_ids is not None
                        else "whole_matrix"
                    ),
                    "ordered_row_ids": batch.ordered_row_ids,
                },
            },
        )

    def execute(row_id: str, resolved: Mapping[str, Any], row_root: Path):
        manifest, path = execute_evaluation_run_spec(
            EvaluationRunSpec.model_validate(resolved),
            registry=registry,
            root=row_root,
            metadata={
                "matrix_harness": {
                    "row_id": row_id,
                    "escape_hatch_reason": escape_hatch_reason,
                }
            },
            execution_context=execution_context,
        )
        return manifest, path

    return MatrixMaterializerHarness(root=root).materialize(
        rows,
        execute=execute,
        command=["python", "-m", "feedbax", "matrix-harness"],
        title="Evaluation run matrix",
        source="EvaluationRunMatrixSpec",
        escape_hatch_reason=escape_hatch_reason,
        matrix_metadata=matrix_metadata,
        regeneration_parameters=regeneration_parameters,
    )


def _execute_evaluation_batch_rows(
    rows: Sequence[tuple[str, Mapping[str, Any]]],
    *,
    registry: EvaluationRecipeRegistry,
    root: Path,
    execution_context: StagedExecutionContext,
    matrix_metadata: Mapping[str, Any],
    regeneration_parameters: Mapping[str, Any],
):
    from feedbax.analysis.harness import MatrixMaterializerHarness
    from feedbax.persistence.manifest_index import connect_index, index_manifest

    if root.exists():
        raise FileExistsError("batched evaluation root must not already exist")
    specs = [(row_id, EvaluationRunSpec.model_validate(resolved)) for row_id, resolved in rows]
    evaluation_types = {spec.evaluation_type for _, spec in specs}
    if len(evaluation_types) != 1:
        raise ValueError("batched evaluation rows must share one evaluation_type")
    evaluation_type = next(iter(evaluation_types))
    batch_recipe = registry.batch(evaluation_type)

    root.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{root.name}.batch-", dir=root.parent))
    try:
        items = [
            EvaluationBatchItem(
                row_id=row_id,
                spec=spec,
            )
            for row_id, spec in specs
        ]
        # Shared-store rows are keyed by content-hash manifest id, so two
        # byte-identical row specs would silently collapse last-write-wins.
        # Fail closed before any recipe execution or publication.
        rows_by_manifest_id: dict[str, list[str]] = {}
        for item in items:
            rows_by_manifest_id.setdefault(evaluation_run_manifest_id(item.spec), []).append(
                item.row_id
            )
        collisions = {
            manifest_id: row_ids
            for manifest_id, row_ids in rows_by_manifest_id.items()
            if len(row_ids) > 1
        }
        if collisions:
            colliding_row_ids = sorted(
                row_id for row_ids in collisions.values() for row_id in row_ids
            )
            raise EvaluationBatchRowError(
                colliding_row_ids[0],
                ValueError(
                    "batch rows share a content-identical evaluation_run_manifest_id "
                    "and would collapse under shared-store publication: "
                    f"{ {manifest_id: sorted(row_ids) for manifest_id, row_ids in collisions.items()} }"
                ),
            )
        for item in items:
            try:
                prerequisites = item.spec.params.get("staged_prerequisites")
                if prerequisites is None:
                    continue
                if not isinstance(prerequisites, Mapping):
                    raise TypeError(
                        "evaluation params.staged_prerequisites must be a mapping or null"
                    )
                for prerequisite in prerequisites.values():
                    resolve_staged_evaluation_prerequisite(
                        prerequisite, execution_context=execution_context
                    )
            except Exception as exc:
                raise EvaluationBatchRowError(item.row_id, exc) from exc
        try:
            results = list(batch_recipe(items, execution_context))
        except EvaluationBatchRowError as exc:
            if exc.row_id not in {item.row_id for item in items}:
                raise ValueError(f"batch recipe named unknown failing row {exc.row_id!r}") from exc
            raise
        if len(results) != len(items) or any(
            not isinstance(result, EvaluationRecipeResult) for result in results
        ):
            raise TypeError("batch recipe must return one EvaluationRecipeResult per row")
        provenance = collect_git_provenance()
        completed: dict[str, tuple[EvaluationRunManifest, Path]] = {}
        batch_metadata = {"row_ids": [item.row_id for item in items]}
        for item, result in zip(items, results, strict=True):
            try:
                completed[item.row_id] = _store_batched_evaluation_result(
                    item,
                    result,
                    store_root=staging,
                    published_root=root,
                    provenance=provenance,
                    metadata={
                        "matrix_harness": {
                            "row_id": item.row_id,
                            "batch_execution": batch_metadata,
                        }
                    },
                )
            except Exception as exc:
                raise EvaluationBatchRowError(item.row_id, exc) from exc

        def execute(row_id: str, _resolved: Mapping[str, Any], _row_root: Path):
            return completed[row_id]

        result = MatrixMaterializerHarness(root=staging, shared_store=True).materialize(
            rows,
            execute=execute,
            command=["python", "-m", "feedbax", "matrix-harness"],
            title="Evaluation run matrix",
            source="EvaluationRunMatrixSpec",
            matrix_metadata=matrix_metadata,
            regeneration_parameters=regeneration_parameters,
        )
        # Publish into one shared physical store: every row's manifest lands in
        # the single manifest tree keyed by its unique manifest id, and one
        # store-wide SQLite index records all rows against their published
        # paths. The index is built once (not once per row), so the fixed store
        # overhead does not scale with row count.
        index = connect_index(staging / "index" / "feedbax.sqlite")
        try:
            published_rows = []
            for row in result.rows:

                def publish(artifact: ArtifactRef) -> ArtifactRef:
                    if artifact.uri is None:
                        return artifact
                    uri = Path(artifact.uri)
                    if not uri.is_relative_to(staging):
                        return artifact
                    return artifact.model_copy(update={"uri": str(root / uri.relative_to(staging))})

                manifest = row.result.model_copy(
                    update={"artifacts": [publish(artifact) for artifact in row.result.artifacts]}
                )
                staged_path = write_manifest(manifest, root=staging, index=False)
                published_path = root / staged_path.relative_to(staging)
                index_manifest(index, manifest, path=published_path)
                published_rows.append(
                    replace(
                        row,
                        result=manifest,
                        manifest_path=published_path,
                        artifacts=tuple(publish(artifact) for artifact in row.artifacts),
                    )
                )
        finally:
            index.close()
        staging.rename(root)
        return replace(result, rows=tuple(published_rows))
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise


def _store_batched_evaluation_result(
    item: EvaluationBatchItem,
    result: EvaluationRecipeResult,
    *,
    store_root: Path,
    published_root: Path,
    provenance: Provenance,
    metadata: Mapping[str, Any],
) -> tuple[EvaluationRunManifest, Path]:
    """Stage one row into the shared batch store using the per-row custody contracts.

    ``store_root`` is the shared staging store that every row is written into;
    ``published_root`` is the shared store's final published location, recorded
    in the row's cache metadata. Rows never receive a private ``row_id`` store
    subtree: state caches, durable state artifacts, and manifests are keyed by
    the row's globally unique manifest identifier, so they coexist in one tree
    without collision.
    """
    manifest_id = evaluation_run_manifest_id(item.spec)
    states_path = evaluation_states_cache_path(manifest_id, root=store_root)
    states_path.parent.mkdir(parents=True, exist_ok=True)
    cache_metadata: dict[str, Any] = {
        "states_path": str(evaluation_states_cache_path(manifest_id, root=published_root)),
        "states_cache_key": manifest_id,
        "states_cache_hit": False,
    }
    if result.states is not None:
        write_evaluation_states_cache(states_path, manifest_id=manifest_id, states=result.states)
        cache_metadata["states_cache_saved"] = True
    if _states_custody_for_spec(item.spec) == "durable":
        result = _with_durable_states_artifact(
            result,
            root=store_root,
            manifest_id=manifest_id,
        )
    prov = provenance.model_copy(deep=True)
    prerequisites = item.spec.params.get("staged_prerequisites") or {}
    prov.parents = [
        *item.spec.inputs,
        *[
            StagedEvaluationPrerequisite.model_validate(value).parent
            for value in prerequisites.values()
        ],
    ]
    prov.entrypoint = EntrypointRef(
        kind="feedbax-evaluation-recipe", name=item.spec.evaluation_type
    )
    manifest = _build_evaluation_manifest(
        manifest_id=manifest_id,
        run_spec=item.spec,
        status="completed",
        provenance=prov,
        summary_metrics={"input_training_runs": len(item.spec.inputs), **result.summary_metrics},
        artifacts=result.artifacts,
        metadata={**result.metadata, **metadata, "cache": cache_metadata},
    )
    return manifest, write_manifest(manifest, root=store_root, index=False)


def resolve_staged_evaluation_prerequisite(
    prerequisite: StagedEvaluationPrerequisite | Mapping[str, Any],
    *,
    execution_context: StagedExecutionContext,
) -> Any:
    """Resolve a v1/v2 staged evaluation-state prerequisite through retained authority.

    Typed v3 containers fail closed with ``EvaluationStatesCustodyUnavailable``.
    """
    declared = StagedEvaluationPrerequisite.model_validate(prerequisite)
    return execution_context.load_evaluation_states(
        declared.parent,
        prerequisite_artifact_provider=declared.artifact_provider,
        validate_staged_prerequisite=True,
    )


def _validate_matrix_staged_parent_references(
    matrix: EvaluationRunMatrixSpec,
    rows: Sequence[MaterializedMatrixRow[EvaluationRunSpec]],
) -> None:
    """Require every row to consume every matrix-level staged parent."""
    for row in rows:
        staged = row.payload.params.get("staged_prerequisites") or {}
        if not isinstance(staged, Mapping):
            raise TypeError("evaluation params.staged_prerequisites must be a mapping or null")
        for name, prerequisite in matrix.staged_parents.items():
            input_match = any(
                not restated_parent_differences(prerequisite.parent, input_ref)
                for input_ref in row.payload.inputs
            )
            stated_prerequisite = (
                StagedEvaluationPrerequisite.model_validate(staged[name])
                if name in staged
                else None
            )
            staged_match = stated_prerequisite is not None and (
                stated_prerequisite.artifact_provider == prerequisite.artifact_provider
                and not restated_parent_differences(prerequisite.parent, stated_prerequisite.parent)
            )
            if not input_match and not staged_match:
                raise ValueError(
                    f"matrix row {row.row_id!r} does not reference staged parent {name!r}"
                )


def _resolve_matrix_staged_parents(
    matrix: EvaluationRunMatrixSpec,
    *,
    parent_manifest_root: Path | str | None,
    execution_descriptor: StagedExecutionDescriptor | Mapping[str, Any] | None,
    artifact_provider_bindings: Sequence[StagedArtifactProviderRootBinding],
    checkpoint_custody_bindings: Sequence[StagedCheckpointCustodyRootBinding],
    execution_context: StagedExecutionContext | None = None,
) -> StagedExecutionContext:
    """Bind and authenticate matrix parents before any row root or recipe effect."""
    if execution_context is not None and (
        execution_descriptor is not None
        or artifact_provider_bindings
        or checkpoint_custody_bindings
    ):
        raise StagedExecutionContextError(
            "pre-resolved execution_context cannot be combined with raw staged bindings"
        )
    context = execution_context or resolve_staged_execution_context(
        execution_descriptor,
        artifact_provider_bindings=artifact_provider_bindings,
        checkpoint_custody_bindings=checkpoint_custody_bindings,
    )
    locations: list[StagedParentExecutionLocation] = []
    for name, prerequisite in matrix.staged_parents.items():
        parent = prerequisite.parent
        if prerequisite.artifact_provider is None:
            bound_manifest_root = context.manifest_roots.get(name)
            if bound_manifest_root is not None:
                candidate_root = bound_manifest_root
            elif parent_manifest_root is not None:
                candidate_root = Path(parent_manifest_root)
            else:
                raise StagedExecutionContextError(
                    f"local matrix staged parent {name!r} requires parent_manifest_root "
                    "or a bound manifest root"
                )
            local_root = Path(candidate_root)
            if not local_root.is_absolute() or ".." in local_root.parts:
                raise StagedExecutionContextError(
                    "matrix parent manifest root must be absolute and contain no '..'"
                )
            try:
                resolved_root = local_root.resolve(strict=True)
            except OSError as exc:
                raise StagedExecutionContextError(
                    f"matrix parent manifest root is unavailable: {local_root}"
                ) from exc
            if resolved_root != local_root:
                raise StagedExecutionContextError(
                    "matrix parent manifest root must not traverse symbolic links"
                )
            if parent.kind == "TrainingRunManifest":
                resolved_training = resolve_evaluation_inputs(
                    EvaluationRunSpec(
                        evaluation_type="feedbax.internal.matrix_parent_preflight",
                        inputs=[parent],
                    ),
                    manifest_root=local_root,
                )[0]
                execution_uri = resolved_training.reference
            else:
                resolved = resolve_manifest_input(parent, local_root)
                execution_uri = resolved.path.relative_to(local_root).as_posix()
            location_root = local_root
        else:
            provider = context.artifact_provider(prerequisite.artifact_provider)
            digest = str(parent.metadata["manifest_sha256"])
            size_bytes = int(parent.metadata["size_bytes"])
            artifact_id = f"artifact://sha256/{digest}"
            execution_uri = provider.canonical_relative_path(
                artifact_id,
                size_bytes=size_bytes,
            ).as_posix()
            location_root = Path(provider.root)
        locations.append(
            StagedParentExecutionLocation(
                parent=parent,
                root=location_root,
                execution_uri=execution_uri,
                artifact_provider=prerequisite.artifact_provider,
            )
        )
    context = with_staged_parent_execution_locations(context, locations)
    for prerequisite in matrix.staged_parents.values():
        context.resolve_manifest_input(prerequisite.parent)
    return context


class EvaluationRecipeExecutionError(RuntimeError):
    """Raised after a registered recipe fails and a failed manifest is written."""

    def __init__(self, manifest: EvaluationRunManifest, path: Path, cause: BaseException):
        super().__init__(
            f"Evaluation recipe for {manifest.id!r} failed; failed manifest written to {path}"
        )
        self.manifest = manifest
        self.path = path
        self.__cause__ = cause


class EvaluationFailureDiagnostics(StrictModel):
    """Opaque schema-identified scientific evidence for a failed recipe."""

    schema_id: str
    schema_version: str
    values: dict[str, Any]

    @model_validator(mode="after")
    def _validate_payload(self) -> "EvaluationFailureDiagnostics":
        validate_schema_identity(self.schema_id, family="evaluation failure diagnostics")
        validate_schema_version(self.schema_version, family="evaluation failure diagnostics")
        try:
            encoded = json.dumps(
                self.model_dump(mode="python"),
                allow_nan=False,
                separators=(",", ":"),
                sort_keys=True,
            ).encode()
        except (TypeError, ValueError) as exc:
            raise ValueError("evaluation failure diagnostics must be finite JSON") from exc
        if len(encoded) > _MAX_FAILURE_DIAGNOSTICS_BYTES:
            raise ValueError("evaluation failure diagnostics exceed the size limit")
        return self


class EvaluationRecipeDiagnosticError(RuntimeError):
    """Recipe failure carrying validated opaque scientific diagnostics."""

    def __init__(
        self,
        message: str,
        diagnostics: EvaluationFailureDiagnostics | Mapping[str, Any],
    ):
        super().__init__(message)
        try:
            self.diagnostics = EvaluationFailureDiagnostics.model_validate(diagnostics)
        except (TypeError, ValueError) as exc:
            raise ValueError("evaluation failure diagnostics payload is invalid") from exc


class EvaluationStatesArtifactNotFound(LookupError):
    """Raised when a manifest has no durable evaluation-states artifact."""


class EvaluationStatesCacheCorruption(RuntimeError):
    """Raised when a states cache payload cannot be trusted."""


def _cache_corruption(path: Path, reason: str) -> EvaluationStatesCacheCorruption:
    return EvaluationStatesCacheCorruption(
        f"Evaluation states cache {path} is corrupt or unsupported: {reason}"
    )


def load_evaluation_states_cache(path: Path, *, manifest_id: str) -> Any:
    """Load a versioned evaluation-states cache payload."""
    try:
        with path.open("rb") as stream:
            payload = pickle.load(stream)
    except (OSError, EOFError, pickle.UnpicklingError, AttributeError, ImportError) as exc:
        raise _cache_corruption(path, str(exc)) from exc

    if not isinstance(payload, Mapping):
        raise _cache_corruption(path, "missing versioned payload envelope")
    schema_version = payload.get("schema_version")
    if schema_version != EVALUATION_STATES_CACHE_SCHEMA_VERSION:
        raise _cache_corruption(
            path,
            f"schema_version={schema_version!r}, "
            f"expected {EVALUATION_STATES_CACHE_SCHEMA_VERSION!r}",
        )
    cached_manifest_id = payload.get("manifest_id")
    if cached_manifest_id != manifest_id:
        raise _cache_corruption(
            path,
            f"manifest_id={cached_manifest_id!r}, expected {manifest_id!r}",
        )
    if "states" not in payload:
        raise _cache_corruption(path, "missing states payload")
    return payload["states"]


def write_evaluation_states_cache(path: Path, *, manifest_id: str, states: Any) -> None:
    """Write a versioned evaluation-states cache payload."""
    with path.open("wb") as stream:
        pickle.dump(
            {
                "schema_version": EVALUATION_STATES_CACHE_SCHEMA_VERSION,
                "manifest_id": manifest_id,
                "states": states,
            },
            stream,
        )


def coerce_evaluation_run_spec(
    value: EvaluationRunSpec | Mapping[str, Any] | Path | str,
) -> EvaluationRunSpec:
    """Load an ``EvaluationRunSpec`` from an object, mapping, or JSON file path.

    Serialized documents are admitted through the registered schema family, so
    an unversioned historical document is accepted as the named v1 baseline
    while an unknown or explicitly rejected version fails closed.
    """
    from feedbax.contracts.migrations import migrate_evaluation_run_spec_payload

    if isinstance(value, EvaluationRunSpec):
        return value
    if isinstance(value, Mapping):
        payload: Mapping[str, Any] = value
    else:
        payload = json.loads(Path(value).read_text(encoding="utf-8"))
        if not isinstance(payload, Mapping):
            raise ValueError("EvaluationRunSpec document must be a JSON object")
    return EvaluationRunSpec.model_validate(migrate_evaluation_run_spec_payload(payload).payload)


def execute_evaluation_run_spec(
    spec: EvaluationRunSpec | Mapping[str, Any] | Path | str,
    *,
    registry: EvaluationRecipeRegistry,
    root: Path | str | None = None,
    repo_root: Path | str | None = None,
    provenance: Provenance | None = None,
    issues: list[str] | None = None,
    metadata: dict[str, Any] | None = None,
    execution_context: StagedExecutionContext = EMPTY_STAGED_EXECUTION_CONTEXT,
    use_cache: bool = True,
    force: bool = False,
) -> tuple[EvaluationRunManifest, Path]:
    """Execute a serialized evaluation spec and write a truthful manifest.

    The evaluated states cache is ephemeral and lives under
    ``{FEEDBAX_RUNS_DIR}/cache/states`` keyed by the deterministic evaluation
    manifest identifier, not by a Studio database hash.
    """
    run_spec = coerce_evaluation_run_spec(spec)
    recipe = registry.get(run_spec.evaluation_type)
    root_path = Path(root) if root is not None else default_manifest_root()
    execution_context = with_staged_repo_root(execution_context, repo_root)
    manifest_id = evaluation_run_manifest_id(run_spec)
    states_path = evaluation_states_cache_path(manifest_id, root=root_path)
    manifest_path = canonical_manifest_path("EvaluationRunManifest", manifest_id, root=root_path)
    states_path.parent.mkdir(parents=True, exist_ok=True)

    prov = provenance.model_copy(deep=True) if provenance is not None else collect_git_provenance()
    staged_prerequisites = run_spec.params.get("staged_prerequisites")
    if staged_prerequisites is None:
        staged_prerequisites = {}
    elif not isinstance(staged_prerequisites, Mapping):
        raise TypeError("evaluation params.staged_prerequisites must be a mapping or null")
    staged_prerequisite_parents = [
        StagedEvaluationPrerequisite.model_validate(value).parent
        for value in staged_prerequisites.values()
    ]
    expected_parents = [*run_spec.inputs, *staged_prerequisite_parents]
    if provenance is not None and prov.parents and prov.parents != expected_parents:
        raise ValueError(
            "evaluation provenance parents disagree with inputs and staged prerequisites"
        )
    prov.parents = expected_parents
    if issues:
        prov.issues.extend(issue for issue in issues if issue not in prov.issues)
    if prov.entrypoint is None:
        prov.entrypoint = EntrypointRef(
            kind="feedbax-evaluation-recipe",
            name=run_spec.evaluation_type,
        )

    manifest_metadata = dict(metadata or {})
    states_custody = _states_custody_for_spec(run_spec)
    cache_metadata: dict[str, Any] = {
        "states_path": str(states_path),
        "states_cache_key": manifest_id,
        "states_cache_hit": False,
    }
    try:
        cache_hit = False
        if use_cache and not force and states_path.exists():
            states = load_evaluation_states_cache(states_path, manifest_id=manifest_id)
            previous_manifest = _load_completed_evaluation_manifest(manifest_path, manifest_id)
            summary_metrics = dict(previous_manifest.summary_metrics) if previous_manifest else {}
            artifacts = list(previous_manifest.artifacts) if previous_manifest else []
            result_metadata = dict(previous_manifest.metadata) if previous_manifest else {}
            result = EvaluationRecipeResult(
                states=states,
                summary_metrics=summary_metrics,
                artifacts=artifacts,
                metadata=result_metadata,
            )
            cache_metadata["states_cache_hit"] = True
            cache_hit = True
        else:
            result = recipe(run_spec, root_path, states_path, execution_context)
            if use_cache and result.states is not None:
                write_evaluation_states_cache(
                    states_path,
                    manifest_id=manifest_id,
                    states=result.states,
                )
                cache_metadata["states_cache_saved"] = True
        if states_custody == "durable":
            result = _with_durable_states_artifact(
                result,
                root=root_path,
                manifest_id=manifest_id,
                allow_existing=cache_hit,
            )

        summary_metrics = {
            "input_training_runs": len(run_spec.inputs),
            **result.summary_metrics,
        }
        manifest = _build_evaluation_manifest(
            manifest_id=manifest_id,
            run_spec=run_spec,
            status="completed",
            provenance=prov,
            summary_metrics=summary_metrics,
            artifacts=result.artifacts,
            metadata={
                **manifest_metadata,
                **result.metadata,
                "cache": cache_metadata,
            },
        )
        return manifest, write_manifest(manifest, root=root_path)
    except Exception as exc:
        error: dict[str, Any] = {
            "type": type(exc).__name__,
            "message": str(exc),
        }
        if isinstance(exc, EvaluationRecipeDiagnosticError):
            error["diagnostics"] = {
                **exc.diagnostics.model_dump(mode="json"),
                "recipe": {
                    "evaluation_type": run_spec.evaluation_type,
                    "entrypoint": EntrypointRef(
                        kind="feedbax-evaluation-recipe",
                        name=run_spec.evaluation_type,
                    ).model_dump(mode="json", exclude_none=True),
                },
            }
        manifest = _build_evaluation_manifest(
            manifest_id=manifest_id,
            run_spec=run_spec,
            status="failed",
            provenance=prov,
            summary_metrics={"input_training_runs": len(run_spec.inputs)},
            artifacts=[],
            metadata={
                **manifest_metadata,
                "cache": cache_metadata,
                "error": error,
            },
        )
        path = write_manifest(manifest, root=root_path)
        raise EvaluationRecipeExecutionError(manifest, path, exc) from exc


def _load_completed_evaluation_manifest(
    path: Path,
    manifest_id: str,
) -> EvaluationRunManifest | None:
    if not path.exists():
        return None
    manifest = load_manifest(path)
    if not isinstance(manifest, EvaluationRunManifest):
        return None
    if manifest.id != manifest_id or manifest.status != "completed":
        return None
    return manifest


def load_evaluation_states(
    manifest: EvaluationRunManifest,
    *,
    root: Path | str | None = None,
) -> Any:
    """Load durable evaluation states from a completed evaluation manifest."""
    artifact = _evaluation_states_artifact(manifest)
    root_path = Path(root) if root is not None else default_manifest_root()
    return load_evaluation_states_artifact(artifact, root=root_path)


def _states_custody_for_spec(run_spec: EvaluationRunSpec) -> str:
    raw = run_spec.params.get("states_custody", "cache")
    if raw not in {"cache", "durable"}:
        raise ValueError(
            f"EvaluationRunSpec params states_custody must be 'cache' or 'durable'; got {raw!r}"
        )
    return str(raw)


def _evaluation_states_artifact(manifest: EvaluationRunManifest) -> ArtifactRef:
    matches = [
        artifact
        for artifact in manifest.artifacts
        if artifact.role == EVALUATION_STATES_ARTIFACT_ROLE
    ]
    if not matches:
        raise EvaluationStatesArtifactNotFound(
            f"Evaluation manifest {manifest.id!r} has no evaluation_states artifact."
        )
    if len(matches) > 1:
        raise ValueError(
            f"Evaluation manifest {manifest.id!r} has multiple evaluation_states artifacts."
        )
    return matches[0]


def _with_durable_states_artifact(
    result: EvaluationRecipeResult,
    *,
    root: Path,
    manifest_id: str,
    allow_existing: bool = False,
) -> EvaluationRecipeResult:
    existing = [
        artifact
        for artifact in result.artifacts
        if artifact.role == EVALUATION_STATES_ARTIFACT_ROLE
    ]
    if existing and allow_existing:
        if len(existing) > 1:
            raise ValueError(
                f"Evaluation manifest {manifest_id!r} has multiple evaluation_states artifacts."
            )
        return result
    if existing:
        raise ValueError(
            "Evaluation recipes must not emit evaluation_states artifacts directly; "
            "durable state custody is handled by execute_evaluation_run_spec."
        )
    artifact = store_evaluation_states_artifact(
        result.states,
        root=root,
        manifest_id=manifest_id,
    )
    return EvaluationRecipeResult(
        states=result.states,
        summary_metrics=result.summary_metrics,
        artifacts=[*result.artifacts, artifact],
        metadata=result.metadata,
    )


def _build_evaluation_manifest(
    *,
    manifest_id: str,
    run_spec: EvaluationRunSpec,
    status: ManifestStatus,
    provenance: Provenance,
    summary_metrics: dict[str, Any],
    artifacts: list[ArtifactRef],
    metadata: dict[str, Any],
) -> EvaluationRunManifest:
    return EvaluationRunManifest(
        id=manifest_id,
        status=status,
        evaluation_spec=spec_payload(
            "EvaluationRunSpec",
            run_spec.model_dump(mode="json", exclude_none=True),
        ),
        input_training_runs=list(run_spec.inputs),
        summary_metrics=summary_metrics,
        provenance=provenance,
        artifacts=artifacts,
        metadata=metadata,
    )
