"""Pydantic models for training specifications."""

from __future__ import annotations

import numbers
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Dict, Generic, List, Literal, Optional, TypeVar

from pydantic import BaseModel, ConfigDict, Field, model_validator

from feedbax.contracts.graph import (
    GraphSpec,
    ParamValue,
    RetentionPolicySpec,
    StudioTaskTimelineSpec,
)
from feedbax.contracts.checkpoints import (
    CheckpointContinuationRequest,
    CheckpointSegmentLineage,
)
from feedbax.contracts.manifest import StrictModel
from feedbax.contracts.worker import (
    AxisSpec,
    CheckpointBarrierSpec,
    CheckpointSlotManifest,
    CheckpointSlotSpec,
    EffectivePhaseSpec,
    MethodContractSpec,
    MappingLevelSpec,
    OptimizerTargetBinding,
    PhaseProgramSpec,
    PhaseSpec,
    ProgressCoordinate,
    ResumeCoordinateSpec,
    StateSlotSpec,
    TrainingBatchProgressSpec,
    UpdateKernelSpec,
    UpdateStepSpec,
    derive_consistency_predicate,
)

if TYPE_CHECKING:
    from feedbax.training.preparation import ExecutionPreparationProvider
    from feedbax.training.row_lowering import TrainingRowLowererRegistration


TRAINING_RUN_SPEC_SCHEMA_ID = "feedbax.spec.training_run"
TRAINING_RUN_SPEC_SCHEMA_VERSION_V1 = "feedbax.spec.training_run.v1"
TRAINING_RUN_SPEC_SCHEMA_VERSION_V2 = "feedbax.spec.training_run.v2"
TRAINING_RUN_SPEC_SCHEMA_VERSION_V3 = "feedbax.spec.training_run.v3"
TRAINING_RUN_SPEC_SCHEMA_VERSION = "feedbax.spec.training_run.v4"
RUN_CONTROL_SPEC_SCHEMA_ID = "feedbax.spec.training.run_control"
RUN_CONTROL_SPEC_SCHEMA_VERSION = "feedbax.spec.training.run_control.v1"
LR_SCHEDULE_SPEC_SCHEMA_ID = "feedbax.spec.training.lr_schedule"
LR_SCHEDULE_SPEC_SCHEMA_VERSION_V1 = "feedbax.spec.training.lr_schedule.v1"
LR_SCHEDULE_SPEC_SCHEMA_VERSION = "feedbax.spec.training.lr_schedule.v2"
LOSS_TERM_SPEC_SCHEMA_ID = "feedbax.spec.training.loss_term"
LOSS_TERM_SPEC_SCHEMA_VERSION_V1 = "feedbax.spec.training.loss_term.v1"
LOSS_TERM_SPEC_SCHEMA_VERSION = "feedbax.spec.training.loss_term.v2"
STANDARD_SUPERVISED_METHOD_REF = "feedbax/standard_supervised/v1"
STANDARD_SUPERVISED_METHOD_PAYLOAD_SCHEMA_ID = (
    "feedbax.spec.training_method.standard_supervised_payload"
)
STANDARD_SUPERVISED_METHOD_PAYLOAD_SCHEMA_VERSION = (
    "feedbax.spec.training_method.standard_supervised_payload.v1"
)
TRAINING_MANIFEST_METADATA_PROJECTION_SCHEMA_ID = (
    "feedbax.spec.training_manifest_metadata_projection"
)
TRAINING_MANIFEST_METADATA_PROJECTION_SCHEMA_VERSION = (
    "feedbax.spec.training_manifest_metadata_projection.v1"
)
SCHEDULE_PROJECTION_SCHEMA_ID = "feedbax.spec.training.schedule_projection"
SCHEDULE_PROJECTION_SCHEMA_VERSION = f"{SCHEDULE_PROJECTION_SCHEMA_ID}.v1"


class BatchScheduleOriginSpec(BaseModel):
    """Typed clock origin for a batch-parameterized schedule.

    This origin selects the training-batch clock for learning-rate and other
    batch schedules. For phase-program step clocks, use
    :class:`feedbax.contracts.worker.ScheduleOriginSpec` instead.
    """

    model_config = ConfigDict(extra="forbid")

    kind: Literal["segment_start", "run_start", "absolute"]
    batch: int | None = Field(default=None, ge=0)

    @model_validator(mode="after")
    def _validate_origin(self) -> "BatchScheduleOriginSpec":
        if self.kind == "absolute" and self.batch is None:
            raise ValueError("/batch is required when schedule origin kind='absolute'")
        if self.kind != "absolute" and self.batch is not None:
            raise ValueError("/batch is only allowed when schedule origin kind='absolute'")
        return self


class LrScheduleSpec(BaseModel):
    """Schema-versioned learning-rate schedule specification.

    Attributes:
        schema_id: Stable schema identity for Feedbax learning-rate schedules.
        schema_version: Version of this schedule payload schema.
        kind: Schedule family. ``"constant"`` holds ``learning_rate_0``;
            ``"warmup_cosine"`` linearly warms from
            ``warmup_init_fraction * learning_rate_0`` over
            ``constant_lr_iterations`` steps, then cosine-anneals to
            ``cosine_annealing_alpha * learning_rate_0`` at ``total_steps``;
            ``"delayed_cosine"`` holds ``learning_rate_0`` for
            ``constant_lr_iterations`` steps, then cosine-anneals to the same
            terminal fraction at ``total_steps``.
        learning_rate_0: Peak learning rate. This maps from rlrmp
            ``controller_lr``.
        total_steps: Origin-relative step at which cosine schedules reach
            their terminal value. Cosine schedules hold that terminal value at
            every later applied coordinate. This maps from rlrmp
            ``n_batches_condition``.
        constant_lr_iterations: Warmup length for ``"warmup_cosine"`` and
            constant-prefix length for ``"delayed_cosine"``. This maps from
            rlrmp ``lr_warmup_batches``.
        warmup_init_fraction: Initial learning-rate fraction for
            ``"warmup_cosine"``. This maps from rlrmp
            ``lr_warmup_init_fraction``.
        cosine_annealing_alpha: Terminal learning-rate fraction for cosine
            schedules. This maps from rlrmp ``lr_cosine_alpha``.
    """

    schema_id: Literal["feedbax.spec.training.lr_schedule"] = LR_SCHEDULE_SPEC_SCHEMA_ID
    schema_version: str = LR_SCHEDULE_SPEC_SCHEMA_VERSION
    origin: BatchScheduleOriginSpec
    allow_inert: bool = False
    kind: Literal["constant", "warmup_cosine", "delayed_cosine"] = "constant"
    learning_rate_0: float = Field(gt=0.0)
    total_steps: int | None = Field(default=None, gt=0)
    constant_lr_iterations: int = Field(default=0, ge=0)
    warmup_init_fraction: float = Field(default=0.0, ge=0.0)
    cosine_annealing_alpha: float = Field(default=0.0, ge=0.0)

    @model_validator(mode="before")
    @classmethod
    def _migrate_v1_origin(cls, value: Any) -> Any:
        if not isinstance(value, Mapping):
            return value
        payload = dict(value)
        if payload.get("schema_version") == LR_SCHEDULE_SPEC_SCHEMA_VERSION_V1:
            payload["schema_version"] = LR_SCHEDULE_SPEC_SCHEMA_VERSION
            payload.setdefault("origin", {"kind": "run_start"})
        elif "schema_version" not in payload and "origin" not in payload:
            # Python construction predates durable serialization. Preserve its
            # historical run-global clock while ensuring emitted v2 payloads
            # always contain the mandatory origin field.
            payload["origin"] = {"kind": "run_start"}
        return payload

    @model_validator(mode="after")
    def _validate_schedule_shape(self) -> "LrScheduleSpec":
        if self.schema_id != LR_SCHEDULE_SPEC_SCHEMA_ID:
            raise ValueError(
                f"/schema_id unsupported LrScheduleSpec schema_id {self.schema_id!r}; "
                f"expected {LR_SCHEDULE_SPEC_SCHEMA_ID!r}"
            )
        if self.schema_version != LR_SCHEDULE_SPEC_SCHEMA_VERSION:
            raise ValueError(
                "/schema_version unsupported LrScheduleSpec schema_version "
                f"{self.schema_version!r}; expected {LR_SCHEDULE_SPEC_SCHEMA_VERSION!r}"
            )
        if self.kind == "constant":
            return self
        if self.total_steps is None:
            raise ValueError(f"/total_steps is required when lr_schedule.kind={self.kind!r}")
        if self.kind == "warmup_cosine":
            if self.constant_lr_iterations < 1:
                raise ValueError("/constant_lr_iterations must be >= 1 for warmup_cosine schedules")
            if self.constant_lr_iterations >= self.total_steps:
                raise ValueError("/constant_lr_iterations must be < /total_steps for warmup_cosine")
        if self.kind == "delayed_cosine" and self.constant_lr_iterations >= self.total_steps:
            raise ValueError("/constant_lr_iterations must be < /total_steps for delayed_cosine")
        return self


class ScheduleProjectionSample(BaseModel):
    """One normalized pre-update schedule value at a global batch coordinate."""

    model_config = ConfigDict(extra="forbid")

    coordinate: int = Field(ge=0)
    value: float = Field(allow_inf_nan=False)


class GovernedScheduleProjection(BaseModel):
    """Sampled values and origin for one stable governed schedule identity."""

    model_config = ConfigDict(extra="forbid")

    origin: BatchScheduleOriginSpec
    samples: list[ScheduleProjectionSample] = Field(min_length=1)

    @model_validator(mode="after")
    def _validate_samples(self) -> "GovernedScheduleProjection":
        coordinates = [sample.coordinate for sample in self.samples]
        if coordinates != sorted(coordinates) or len(coordinates) != len(set(coordinates)):
            raise ValueError("schedule projection coordinates must be unique and sorted")
        return self


class ScheduleProjection(BaseModel):
    """Complete versioned evaluation table for all governed schedules."""

    model_config = ConfigDict(extra="forbid")

    schema_id: Literal["feedbax.spec.training.schedule_projection"] = SCHEDULE_PROJECTION_SCHEMA_ID
    schema_version: Literal["feedbax.spec.training.schedule_projection.v1"] = (
        SCHEDULE_PROJECTION_SCHEMA_VERSION
    )
    complete: Literal[True] = True
    schedules: dict[str, GovernedScheduleProjection] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _validate_inventory(self) -> "ScheduleProjection":
        if any(not schedule_id for schedule_id in self.schedules):
            raise ValueError("schedule projection IDs must be non-empty")
        coordinate_sets = {
            tuple(sample.coordinate for sample in schedule.samples)
            for schedule in self.schedules.values()
        }
        if len(coordinate_sets) > 1:
            raise ValueError("all projected schedules must use the same coordinates")
        return self


class OptimizerSpec(BaseModel):
    """Specification for an optimizer.

    Attributes:
        type: Optimizer factory name. The public builder supports ``"adamw"``,
            ``"adam"``, ``"sgd"``, and ``"rmsprop"``.
        params: Static optimizer parameters other than scheduled learning rate.
        lr_schedule: Optional declarative learning-rate schedule. ``None``
            preserves legacy method-owned optimizer construction.
    """

    type: str
    params: Dict[str, ParamValue] = Field(default_factory=dict)
    lr_schedule: LrScheduleSpec | None = None


class TimeAggregationSpec(BaseModel):
    """Specification for time aggregation in loss computation."""

    mode: Literal["all", "mean", "sum", "final", "range", "segment", "custom"] = "all"
    start: Optional[int] = None
    end: Optional[int] = None
    segment_name: Optional[str] = None
    time_idxs: Optional[List[int]] = None
    discount: Optional[Literal["none", "power", "linear"]] = None
    discount_exp: Optional[float] = None


class LossTermSpec(BaseModel):
    """Specification for a loss term."""

    schema_id: Literal["feedbax.spec.training.loss_term"] = LOSS_TERM_SPEC_SCHEMA_ID
    schema_version: str = LOSS_TERM_SPEC_SCHEMA_VERSION
    type: str
    label: str
    weight: float = 1.0
    selector: Optional[str] = None
    target_selector: Optional[str] = None
    target_value: Optional[Any] = None
    retention: Optional[RetentionPolicySpec] = None
    norm: Optional[Literal["squared_l2", "l2", "l1", "huber"]] = None
    matrix: Optional[Any] = None
    matrix_kind: Optional[Literal["dense", "diagonal"]] = None
    time_agg: Optional[TimeAggregationSpec] = None
    children: Optional[Dict[str, "LossTermSpec"]] = None


class EarlyStoppingSpec(BaseModel):
    """Specification for early stopping."""

    metric: str
    patience: int
    min_delta: float


class TrainingSpec(BaseModel):
    """Complete specification for a training run."""

    optimizer: OptimizerSpec
    loss: LossTermSpec
    n_batches: int
    batch_size: int
    n_epochs: Optional[int] = None
    checkpoint_interval: Optional[int] = None
    early_stopping: Optional[EarlyStoppingSpec] = None


class TaskSpec(BaseModel):
    """Specification for a task."""

    type: str
    params: Dict[str, ParamValue] = Field(default_factory=dict)
    timeline: Optional[StudioTaskTimelineSpec] = None


class TrainingConfig(BaseModel):
    """Structured configuration for the real JAX training backend (Phase 6).

    Passed verbatim to the worker via the ``/start`` request body under the
    ``training_config`` key.  The worker converts this into a ``_TrainingCfg``
    dataclass via ``_extract_training_cfg``.  All fields have sensible defaults
    so callers can override only the parameters they care about.

    Attributes:
        n_batches: Number of training steps.
        batch_size: Trials per gradient update. The current generic graph worker
            supports only ``batch_size=1`` and rejects larger values at compile
            time instead of silently ignoring them.
        learning_rate: AdamW learning rate.
        grad_clip: Global gradient clipping norm. The schema default remains 1.0;
            set explicitly to None to disable gradient clipping.
        hidden_dim: GRU / CDE hidden state dimension.
        network_type: Controller architecture — ``"gru"`` or ``"cde"``.
        n_reach_steps: Number of control steps per episode.
        effort_weight: Relative weight for the muscle-effort penalty.
        snapshot_interval: Emit a ``training_trajectory`` event every N batches.
    """

    n_batches: int = 2000
    batch_size: int = 128
    learning_rate: float = 1e-3
    grad_clip: float | None = 1.0
    hidden_dim: int = 128
    network_type: str = "gru"
    n_reach_steps: int = 80
    effort_weight: float = 2.5
    snapshot_interval: int = 100


class TrainingRunContractModel(StrictModel):
    """Base model for durable training-run request contracts."""

    model_config = ConfigDict(extra="forbid")


class RunControlSpec(TrainingRunContractModel):
    """Small, method-agnostic control surface for one training segment."""

    model_config = ConfigDict(extra="forbid", strict=True)

    schema_id: Literal["feedbax.spec.training.run_control"] = RUN_CONTROL_SPEC_SCHEMA_ID
    schema_version: str = RUN_CONTROL_SPEC_SCHEMA_VERSION
    n_batches: int = Field(gt=0)
    batch_size: int = Field(gt=0)
    checkpoint_interval: int | None = Field(default=None, gt=0)
    progress_interval: int | None = Field(default=None, gt=0)
    continuation: CheckpointContinuationRequest | None = None

    @model_validator(mode="after")
    def _validate_control(self) -> "RunControlSpec":
        if self.schema_version != RUN_CONTROL_SPEC_SCHEMA_VERSION:
            raise ValueError(
                "/schema_version unsupported RunControlSpec schema_version "
                f"{self.schema_version!r}; expected {RUN_CONTROL_SPEC_SCHEMA_VERSION!r}; "
                "migration_intentionally_absent=yes"
            )
        if self.continuation is not None and self.continuation.additional_batches != self.n_batches:
            raise ValueError(
                "/n_batches must equal /continuation/additional_batches for a continuation"
            )
        return self


class MethodRefSpec(TrainingRunContractModel):
    """Durable namespaced/versioned method identity."""

    package: str = Field(min_length=1)
    name: str = Field(min_length=1)
    version: str = Field(min_length=1)

    @property
    def key(self) -> str:
        """Return the registry key for this method reference."""
        return f"{self.package}/{self.name}/{self.version}"

    @model_validator(mode="after")
    def _validate_segments(self) -> "MethodRefSpec":
        for path, value in (
            ("/method_ref/package", self.package),
            ("/method_ref/name", self.name),
            ("/method_ref/version", self.version),
        ):
            if "/" in value:
                raise ValueError(f"{path} must not contain '/'")
        return self


class GraphTopologySourceSpec(TrainingRunContractModel):
    """GraphSpec source for the run topology."""

    kind: Literal["GraphSpec"] = "GraphSpec"
    ref: str | None = None
    inline: dict[str, Any] | None = None
    schema_id: str | None = None
    schema_version: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _validate_source(self) -> "GraphTopologySourceSpec":
        if self.ref is None and self.inline is None:
            raise ValueError("/graph requires ref or inline GraphSpec payload")
        if self.inline is not None:
            GraphSpec.model_validate(self.inline)
        return self


class ObjectiveSlotSpec(TrainingRunContractModel):
    """Objective/loss payload slot consumed by future objective-governance work."""

    kind: Literal["loss_term", "objective_spec", "external"] = "loss_term"
    loss: LossTermSpec | None = None
    payload: dict[str, Any] | None = None
    schema_id: str | None = None
    schema_version: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _validate_payload(self) -> "ObjectiveSlotSpec":
        if self.kind == "loss_term" and self.loss is None:
            raise ValueError("/objective/loss is required when objective.kind='loss_term'")
        if self.kind != "loss_term" and self.payload is None:
            raise ValueError(f"/objective/payload is required when objective.kind={self.kind!r}")
        return self


class RiskAggregationSpec(TrainingRunContractModel):
    """Objective/risk aggregation policy, separate from method identity."""

    realization: Literal["none", "mean", "sum", "min", "max", "custom"] = "none"
    replicate: Literal["none", "mean", "sum", "min", "max", "custom"] = "none"
    time: TimeAggregationSpec | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class MethodPayloadEnvelope(TrainingRunContractModel):
    """Governed method-owned payload plus schema identity."""

    schema_id: str
    schema_version: str
    payload: dict[str, Any] = Field(default_factory=dict)
    metadata: dict[str, Any] = Field(default_factory=dict)


class MethodExtensionsSpec(TrainingRunContractModel):
    """Explicitly non-semantic method extension metadata."""

    metadata: dict[str, Any] = Field(default_factory=dict)


class TrainingMethodAuthoringContribution(BaseModel):
    """Runtime-only method values consumed by the typed authoring compiler.

    This hook result is not a durable spec. The authoring compiler embeds its
    values in the existing ``TrainingRunSpec`` contract.
    """

    model_config = ConfigDict(extra="forbid", strict=True)

    training_config: TrainingConfig
    checkpoint_interval: int | None = Field(default=None, gt=0)
    progress_interval: int | None = Field(default=None, gt=0)
    method_extensions: MethodExtensionsSpec = Field(default_factory=MethodExtensionsSpec)
    mapping_levels: list[MappingLevelSpec] | None = None


class StandardSupervisedMethodPayload(TrainingRunContractModel):
    """Payload owned by Feedbax's standard supervised training method."""

    optimizer: OptimizerSpec = Field(default_factory=lambda: OptimizerSpec(type="adamw"))
    gradient_clip: float | None = None
    model_slot: str = "model"
    optimizer_slot: str = "optimizer"
    objective_slot: str = "objective"
    metadata: dict[str, Any] = Field(default_factory=dict)


class WorkerExecutionSpec(TrainingRunContractModel):
    """Worker declarations required before launch."""

    method_contract: MethodContractSpec
    effective_phase: EffectivePhaseSpec
    mapping_levels: list[MappingLevelSpec] | None = None
    checkpoint_slots: CheckpointSlotManifest | None = None
    resume: ResumeCoordinateSpec | None = None
    progress: ProgressCoordinate | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class ExecutionPolicySpec(TrainingRunContractModel):
    """Execution policy for a request, without performing launch side effects."""

    mode: Literal["dry_run", "local", "remote"] = "local"
    require_review: bool = True
    allow_cloud: bool = False
    max_wallclock_seconds: int | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class ArtifactPolicySpec(TrainingRunContractModel):
    """Artifact custody and manifest emission policy for a training request."""

    manifest_root: str | None = None
    artifact_root: str | None = None
    custody: Literal["local", "mandible", "external"] = "local"
    metadata: dict[str, Any] = Field(default_factory=dict)


class CheckpointProgressPolicySpec(TrainingRunContractModel):
    """Checkpoint and progress policy for worker-visible run records."""

    checkpoint_interval: int | None = None
    progress_interval: int | None = None
    resume_from: ResumeCoordinateSpec | None = None
    checkpoint_slots: CheckpointSlotManifest | None = None
    continuation: CheckpointContinuationRequest | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


@dataclass(frozen=True)
class TrainingMethodRegistration:
    """Low-level method registration retained for direct runtime adapters."""

    method_ref: str
    payload_schema_id: str
    payload_schema_version: str
    payload_model: type[BaseModel]
    contract_factory: Callable[[], MethodContractSpec] | None
    update_kernels_factory: Callable[[BaseModel], Mapping[str, Callable[..., Mapping[str, Any]]]]
    guard_predicates_factory: Callable[
        [BaseModel], Mapping[str, Callable[..., Mapping[str, Any]]]
    ] = lambda _payload: {}
    rejected_payload_versions: tuple[str, ...] = ()
    owner: str = "feedbax"
    package: str | None = None
    requires_execution_preparation: bool = False
    contract_compiler: Callable[[BaseModel], MethodContractSpec] | None = None

    def compile_contract(self, payload: BaseModel) -> MethodContractSpec:
        """Compile the worker contract, preserving legacy zero-argument factories."""
        if self.contract_compiler is not None:
            return self.contract_compiler(payload)
        if self.contract_factory is None:
            raise ValueError(
                f"training method {self.method_ref!r} has no payload-aware contract compiler"
            )
        return self.contract_factory()


PayloadT = TypeVar("PayloadT", bound=BaseModel)

TrainingMethodOptimizerSpecProjector = Callable[[PayloadT], OptimizerSpec | Mapping[str, Any]]
TrainingMethodOptimizerStepExtractor = Callable[[PayloadT, Mapping[str, Any]], int]


@dataclass(frozen=True)
class TrainingMethodAuthoringHook(Generic[PayloadT]):
    """Complete runtime hook for method-owned authoring compilation."""

    lowerer_id: str
    lowerer_version: str
    compile: Callable[[PayloadT], TrainingMethodAuthoringContribution]
    graph: Callable[[PayloadT], object]
    task: Callable[[PayloadT], object]
    objective: Callable[[PayloadT], object]
    domain: Callable[[PayloadT], Mapping[str, Any]]

    def validate_structure(self) -> None:
        """Validate the stable lowerer identity and callable boundaries."""
        identities = (self.lowerer_id, self.lowerer_version)
        if any(not isinstance(value, str) or not value.strip() for value in identities):
            raise ValueError("training method authoring hook identity must not be empty")
        invalid = [
            name
            for name in ("compile", "graph", "task", "objective", "domain")
            if not callable(getattr(self, name))
        ]
        if invalid:
            raise TypeError(f"training method authoring hook callables are invalid={invalid!r}")


@dataclass(frozen=True)
class TrainingMethodMetadataProjector(Generic[PayloadT]):
    """Typed runtime hook for projecting method-owned manifest metadata."""

    schema_id: str
    schema_version: str
    output_model: type[BaseModel]
    projector: Callable[[PayloadT], object]

    def validate_structure(self) -> None:
        """Validate the stable projection identity and callable boundary."""
        identities = (self.schema_id, self.schema_version)
        if any(not isinstance(value, str) or not value.strip() for value in identities):
            raise ValueError("training method metadata projector identity must not be empty")
        if not isinstance(self.output_model, type) or not issubclass(self.output_model, BaseModel):
            raise TypeError("training method metadata projector output_model must extend BaseModel")
        if self.output_model.model_config.get("extra") != "forbid":
            raise ValueError(
                "training method metadata projector output_model must set extra='forbid'"
            )
        if self.output_model.model_config.get("strict") is not True:
            raise ValueError("training method metadata projector output_model must set strict=True")
        if not callable(self.projector):
            raise TypeError("training method metadata projector projector must be callable")

    def project(self, payload: PayloadT) -> BaseModel:
        """Project and validate metadata through the declared output model."""
        self.validate_structure()
        return self.output_model.model_validate(self.projector(payload))


@dataclass(frozen=True)
class TrainingMethodScheduleProjector(Generic[PayloadT]):
    """Stable runtime hook that evaluates every method-owned batch schedule."""

    projector_id: str
    projector_version: str
    projector: (
        Callable[[PayloadT, Sequence[int]], ScheduleProjection | Mapping[str, Any]] | None
    ) = None
    lineage_projector: (
        Callable[
            [PayloadT, Sequence[int], CheckpointSegmentLineage],
            ScheduleProjection | Mapping[str, Any],
        ]
        | None
    ) = None

    def validate_structure(self) -> None:
        """Validate the stable projector identity and callable boundary."""
        identities = (self.projector_id, self.projector_version)
        if any(not isinstance(value, str) or not value.strip() for value in identities):
            raise ValueError("training method schedule projector identity must not be empty")
        configured = tuple(
            name
            for name, value in (
                ("projector", self.projector),
                ("lineage_projector", self.lineage_projector),
            )
            if value is not None
        )
        if len(configured) != 1:
            raise ValueError(
                "training method schedule projector must configure exactly one of "
                "projector or lineage_projector"
            )
        value = self.projector or self.lineage_projector
        if not callable(value):
            raise TypeError(f"training method schedule projector {configured[0]} must be callable")

    def project(
        self,
        payload: PayloadT,
        coordinates: Sequence[int],
        *,
        lineage: CheckpointSegmentLineage | None = None,
    ) -> ScheduleProjection:
        """Evaluate method schedules with optional authenticated segment lineage."""
        self.validate_structure()
        if self.lineage_projector is None:
            assert self.projector is not None
            raw_projection = self.projector(payload, coordinates)
        else:
            if lineage is None:
                raise ValueError(
                    "lineage-aware training method schedule projection requires "
                    "CheckpointSegmentLineage"
                )
            raw_projection = self.lineage_projector(payload, coordinates, lineage)
        projection = ScheduleProjection.model_validate(raw_projection)
        expected = tuple(int(coordinate) for coordinate in coordinates)
        for schedule_id, schedule in projection.schedules.items():
            actual = tuple(sample.coordinate for sample in schedule.samples)
            if actual != expected:
                raise ValueError(
                    f"method schedule {schedule_id!r} projected coordinates={actual!r}; "
                    f"expected={expected!r}"
                )
        return projection


@dataclass(frozen=True)
class TrainingProgramDeclaration(Generic[PayloadT]):
    """Neutral identity and canonical payload schema for one training program."""

    method_ref: str
    payload_schema_id: str
    payload_schema_version: str
    payload_model: type[PayloadT]
    rejected_payload_versions: tuple[str, ...] = ()
    owner: str = "feedbax"
    package: str | None = None


@dataclass(frozen=True)
class TrainingProgramRuntimeFacet(Generic[PayloadT]):
    """Runtime-owned method contract and numerical kernel factories."""

    contract_compiler: Callable[[PayloadT], MethodContractSpec]
    update_kernels_factory: Callable[[PayloadT], Mapping[str, Callable[..., Mapping[str, Any]]]]
    guard_predicates_factory: Callable[
        [PayloadT], Mapping[str, Callable[..., Mapping[str, Any]]]
    ] = lambda _payload: {}


@dataclass(frozen=True)
class TrainingProgramAuthoringFacet(Generic[PayloadT]):
    """Optional authoring operations paid for only by authorable programs."""

    hook: TrainingMethodAuthoringHook[PayloadT]


@dataclass(frozen=True)
class TrainingProgramPreparationFacet:
    """Runtime preparation supplied independently of method semantics."""

    provider: ExecutionPreparationProvider


@dataclass(frozen=True)
class TrainingProgramRowLoweringFacet:
    """Authored-row compilers derived into the application lowering view."""

    registrations: tuple[TrainingRowLowererRegistration, ...]


@dataclass(frozen=True)
class TrainingProgramProjectionFacet(Generic[PayloadT]):
    """Optional manifest, schedule, and optimizer projections."""

    metadata: TrainingMethodMetadataProjector[PayloadT] | None = None
    schedule: TrainingMethodScheduleProjector[PayloadT] | None = None
    optimizer_spec: TrainingMethodOptimizerSpecProjector[PayloadT] | None = None
    optimizer_step: TrainingMethodOptimizerStepExtractor[PayloadT] | None = None


@dataclass(frozen=True)
class DeclaredTrainingProgram(Generic[PayloadT]):
    """Application-root projection of one declaration and its selected facets."""

    declaration: TrainingProgramDeclaration[PayloadT]
    runtime: TrainingProgramRuntimeFacet[PayloadT]
    preparation: TrainingProgramPreparationFacet | None = None
    row_lowering: TrainingProgramRowLoweringFacet | None = None
    authoring: TrainingProgramAuthoringFacet[PayloadT] | None = None
    projection: TrainingProgramProjectionFacet[PayloadT] | None = None

    @property
    def method_ref(self) -> str:
        return self.declaration.method_ref

    @property
    def payload_schema_id(self) -> str:
        return self.declaration.payload_schema_id

    @property
    def payload_schema_version(self) -> str:
        return self.declaration.payload_schema_version

    @property
    def payload_model(self) -> type[PayloadT]:
        return self.declaration.payload_model

    @property
    def rejected_payload_versions(self) -> tuple[str, ...]:
        return self.declaration.rejected_payload_versions

    @property
    def owner(self) -> str:
        return self.declaration.owner

    @property
    def package(self) -> str | None:
        return self.declaration.package

    @property
    def contract_compiler(self):
        return self.runtime.contract_compiler

    @property
    def update_kernels_factory(self):
        return self.runtime.update_kernels_factory

    @property
    def guard_predicates_factory(self):
        return self.runtime.guard_predicates_factory

    @property
    def preparation_provider(self):
        return self.preparation.provider if self.preparation is not None else None

    @property
    def authoring_hook(self):
        return self.authoring.hook if self.authoring is not None else None

    @property
    def metadata_projector(self):
        return self.projection.metadata if self.projection is not None else None

    @property
    def schedule_projector(self):
        return self.projection.schedule if self.projection is not None else None

    @property
    def optimizer_spec_projector(self):
        return self.projection.optimizer_spec if self.projection is not None else None

    @property
    def optimizer_step_extractor(self):
        return self.projection.optimizer_step if self.projection is not None else None

    def registration(self) -> TrainingMethodRegistration:
        """Derive the runtime registry row from neutral and runtime facets."""
        return TrainingMethodRegistration(
            method_ref=self.method_ref,
            payload_schema_id=self.payload_schema_id,
            payload_schema_version=self.payload_schema_version,
            payload_model=self.payload_model,
            contract_factory=None,
            contract_compiler=self.contract_compiler,
            update_kernels_factory=self.update_kernels_factory,
            guard_predicates_factory=self.guard_predicates_factory,
            rejected_payload_versions=self.rejected_payload_versions,
            owner=self.owner,
            package=self.package,
            requires_execution_preparation=self.preparation is not None,
        )


def declare_training_program(
    *,
    method_ref: str,
    payload_schema_id: str,
    payload_schema_version: str,
    payload_model: type[PayloadT],
    contract_compiler: Callable[[PayloadT], MethodContractSpec],
    update_kernels_factory: Callable[[PayloadT], Mapping[str, Callable[..., Mapping[str, Any]]]],
    guard_predicates_factory: Callable[
        [PayloadT], Mapping[str, Callable[..., Mapping[str, Any]]]
    ] = lambda _payload: {},
    preparation_provider: ExecutionPreparationProvider | None = None,
    row_lowerers: Sequence[TrainingRowLowererRegistration] = (),
    authoring_hook: TrainingMethodAuthoringHook[PayloadT] | None = None,
    metadata_projector: TrainingMethodMetadataProjector[PayloadT] | None = None,
    schedule_projector: TrainingMethodScheduleProjector[PayloadT] | None = None,
    optimizer_spec_projector: TrainingMethodOptimizerSpecProjector[PayloadT] | None = None,
    optimizer_step_extractor: TrainingMethodOptimizerStepExtractor[PayloadT] | None = None,
    rejected_payload_versions: tuple[str, ...] = (),
    owner: str = "feedbax",
    package: str | None = None,
) -> DeclaredTrainingProgram[PayloadT]:
    """Compose a training program from the facets its application needs."""
    projection = None
    if any(
        value is not None
        for value in (
            metadata_projector,
            schedule_projector,
            optimizer_spec_projector,
            optimizer_step_extractor,
        )
    ):
        projection = TrainingProgramProjectionFacet(
            metadata_projector,
            schedule_projector,
            optimizer_spec_projector,
            optimizer_step_extractor,
        )
    return DeclaredTrainingProgram(
        declaration=TrainingProgramDeclaration(
            method_ref=method_ref,
            payload_schema_id=payload_schema_id,
            payload_schema_version=payload_schema_version,
            payload_model=payload_model,
            rejected_payload_versions=rejected_payload_versions,
            owner=owner,
            package=package,
        ),
        runtime=TrainingProgramRuntimeFacet(
            contract_compiler,
            update_kernels_factory,
            guard_predicates_factory,
        ),
        preparation=(
            TrainingProgramPreparationFacet(preparation_provider)
            if preparation_provider is not None
            else None
        ),
        row_lowering=(
            TrainingProgramRowLoweringFacet(tuple(row_lowerers)) if row_lowerers else None
        ),
        authoring=(TrainingProgramAuthoringFacet(authoring_hook) if authoring_hook else None),
        projection=projection,
    )


def evolve_training_program(
    program: DeclaredTrainingProgram[PayloadT],
    **changes: Any,
) -> DeclaredTrainingProgram[PayloadT]:
    """Create a revised declaration while preserving its unmodified facets."""
    values = {
        "method_ref": program.method_ref,
        "payload_schema_id": program.payload_schema_id,
        "payload_schema_version": program.payload_schema_version,
        "payload_model": program.payload_model,
        "contract_compiler": program.contract_compiler,
        "update_kernels_factory": program.update_kernels_factory,
        "guard_predicates_factory": program.guard_predicates_factory,
        "preparation_provider": program.preparation_provider,
        "row_lowerers": (
            program.row_lowering.registrations if program.row_lowering is not None else ()
        ),
        "authoring_hook": program.authoring_hook,
        "metadata_projector": program.metadata_projector,
        "schedule_projector": program.schedule_projector,
        "optimizer_spec_projector": program.optimizer_spec_projector,
        "optimizer_step_extractor": program.optimizer_step_extractor,
        "rejected_payload_versions": program.rejected_payload_versions,
        "owner": program.owner,
        "package": program.package,
    }
    unknown = sorted(set(changes) - set(values))
    if unknown:
        raise TypeError(f"unknown training-program declaration fields={unknown!r}")
    values.update(changes)
    return declare_training_program(**values)


@dataclass(frozen=True)
class ResolvedTrainingMethod(Generic[PayloadT]):
    """Typed, identity-checked runtime projection of one method request."""

    program: DeclaredTrainingProgram[PayloadT] | None
    registration: TrainingMethodRegistration
    payload: PayloadT
    contract: MethodContractSpec
    effective_phase: EffectivePhaseSpec
    update_kernels: Mapping[str, Callable[..., Mapping[str, Any]]] = field(compare=False)
    guard_predicates: Mapping[str, Callable[..., Mapping[str, Any]]] = field(compare=False)


class TrainingManifestMetadataProjection(TrainingRunContractModel):
    """Governed request to project selected external payload metadata."""

    schema_id: Literal["feedbax.spec.training_manifest_metadata_projection"] = (
        TRAINING_MANIFEST_METADATA_PROJECTION_SCHEMA_ID
    )
    schema_version: str = TRAINING_MANIFEST_METADATA_PROJECTION_SCHEMA_VERSION
    source_payload_kind: str = Field(min_length=1)
    source_payload_schema_id: str = Field(min_length=1)
    source_payload_schema_version: str = Field(min_length=1)
    source_payload_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    projection_schema_id: str = Field(min_length=1)
    projection_schema_version: str = Field(min_length=1)
    values: dict[str, Any]

    @model_validator(mode="after")
    def _validate_schema(self) -> "TrainingManifestMetadataProjection":
        if self.schema_version != TRAINING_MANIFEST_METADATA_PROJECTION_SCHEMA_VERSION:
            raise ValueError(
                "/manifest_metadata_projection/schema_version unsupported version "
                f"{self.schema_version!r}; expected "
                f"{TRAINING_MANIFEST_METADATA_PROJECTION_SCHEMA_VERSION!r}; "
                "migration_intentionally_absent=yes"
            )
        if not self.values:
            raise ValueError("/manifest_metadata_projection/values must not be empty")
        if any(not key for key in self.values):
            raise ValueError("/manifest_metadata_projection/values keys must be non-empty strings")
        return self


@dataclass(frozen=True)
class TrainingManifestMetadataProjectionRegistration:
    """Governance row for one external payload metadata projection schema."""

    source_payload_kind: str
    source_payload_schema_id: str
    source_payload_schema_version: str
    projection_schema_id: str
    projection_schema_version: str
    values_model: type[BaseModel]
    owner: str
    package: str

    @property
    def source_key(self) -> tuple[str, str, str]:
        """Return the external training payload identity governed by this row."""
        return (
            self.source_payload_kind,
            self.source_payload_schema_id,
            self.source_payload_schema_version,
        )


class TrainingProgramRegistry:
    """Registry for method payloads and independent manifest projection governance."""

    def __init__(self) -> None:
        self._sealed = False
        self._registrations: dict[str, TrainingMethodRegistration] = {}
        self._programs: dict[str, DeclaredTrainingProgram[Any]] = {}
        self._metadata_projection_registrations: dict[
            tuple[str, str, str], TrainingManifestMetadataProjectionRegistration
        ] = {}

    def register(self, registration: TrainingMethodRegistration) -> None:
        """Register one method payload governance row."""
        self._require_mutable()
        if registration.method_ref in self._registrations:
            raise ValueError(f"training method already registered: {registration.method_ref!r}")
        producer_count = sum(
            producer is not None
            for producer in (registration.contract_factory, registration.contract_compiler)
        )
        if producer_count != 1:
            raise ValueError(
                f"training method {registration.method_ref!r} must define exactly one contract "
                "producer: contract_factory or contract_compiler"
            )
        self._registrations[registration.method_ref] = registration

    def register_program(self, program: DeclaredTrainingProgram[Any]) -> None:
        """Atomically register one declaration and its runtime projection."""
        self._require_mutable()
        required = {
            "method_ref": program.method_ref,
            "payload_schema_id": program.payload_schema_id,
            "payload_schema_version": program.payload_schema_version,
            "owner": program.owner,
        }
        missing = [name for name, value in required.items() if not value]
        if missing:
            raise ValueError(f"training program declaration has empty fields={missing!r}")
        if program.method_ref in self._programs:
            raise ValueError(f"training program already registered: {program.method_ref!r}")
        if not issubclass(program.payload_model, BaseModel):
            raise TypeError("training program payload_model must extend BaseModel")
        hooks = {
            "contract_compiler": program.contract_compiler,
            "update_kernels_factory": program.update_kernels_factory,
            "guard_predicates_factory": program.guard_predicates_factory,
            "preparation_provider": program.preparation_provider,
            "optimizer_spec_projector": program.optimizer_spec_projector,
            "optimizer_step_extractor": program.optimizer_step_extractor,
        }
        invalid_hooks = [
            name for name, hook in hooks.items() if hook is not None and not callable(hook)
        ]
        if invalid_hooks:
            raise TypeError(f"training program has non-callable facets={invalid_hooks!r}")
        if program.row_lowering is not None:
            from feedbax.training.row_lowering import TrainingRowLowererRegistration

            if not program.row_lowering.registrations or any(
                not isinstance(registration, TrainingRowLowererRegistration)
                for registration in program.row_lowering.registrations
            ):
                raise TypeError(
                    "training-program row-lowering facet must contain "
                    "TrainingRowLowererRegistration values"
                )
        if program.authoring_hook is not None:
            if not isinstance(program.authoring_hook, TrainingMethodAuthoringHook):
                raise TypeError(
                    "training-program authoring facet must contain a TrainingMethodAuthoringHook"
                )
            program.authoring_hook.validate_structure()
        if program.metadata_projector is not None:
            if not isinstance(program.metadata_projector, TrainingMethodMetadataProjector):
                raise TypeError(
                    "training-program projection facet metadata must be a "
                    "TrainingMethodMetadataProjector"
                )
            program.metadata_projector.validate_structure()
        if program.schedule_projector is not None:
            if not isinstance(program.schedule_projector, TrainingMethodScheduleProjector):
                raise TypeError(
                    "training-program projection facet schedule must be a "
                    "TrainingMethodScheduleProjector"
                )
            program.schedule_projector.validate_structure()
        self.register(program.registration())
        self._programs[program.method_ref] = program

    def program_keys(self) -> tuple[str, ...]:
        """Return method refs backed by composed declarations."""
        return tuple(sorted(self._programs))

    def program(self, method_ref: MethodRefSpec | str) -> DeclaredTrainingProgram[Any] | None:
        """Return the composed program for a method ref, when declared."""
        key = method_ref.key if isinstance(method_ref, MethodRefSpec) else method_ref
        return self._programs.get(key)

    def available_keys(self) -> tuple[str, ...]:
        """Return method refs known to this registry."""
        return tuple(sorted(self._registrations))

    def resolve(self, method_ref: MethodRefSpec | str, *, path: str) -> TrainingMethodRegistration:
        """Resolve a method ref or raise with available registry keys."""
        key = method_ref.key if isinstance(method_ref, MethodRefSpec) else method_ref
        try:
            return self._registrations[key]
        except KeyError as exc:
            raise ValueError(
                f"{path}: unknown method_ref {key!r}; "
                f"available registry keys={list(self.available_keys())!r}. "
                "Install a package exposing a feedbax.plugins training-method hook, "
                "or pass --plugin <module> to CLI commands that validate "
                "TrainingRunSpec payloads."
            ) from exc

    def validate_payload(
        self,
        method_ref: MethodRefSpec,
        envelope: MethodPayloadEnvelope,
        *,
        path: str,
    ) -> BaseModel:
        """Validate a method-owned payload through its registered schema row."""
        registration = self.resolve(method_ref, path="/method_ref")
        if envelope.schema_id != registration.payload_schema_id:
            raise ValueError(
                f"{path}/schema_id: unsupported method payload schema_id "
                f"{envelope.schema_id!r}; expected {registration.payload_schema_id!r}; "
                f"available registry keys={list(self.available_keys())!r}"
            )
        if envelope.schema_version != registration.payload_schema_version:
            if envelope.schema_version in registration.rejected_payload_versions:
                raise ValueError(
                    f"{path}/schema_version: unsupported method payload schema version "
                    f"{envelope.schema_version!r}; current_version="
                    f"{registration.payload_schema_version!r}; "
                    "migration_intentionally_absent=yes; "
                    f"available registry keys={list(self.available_keys())!r}"
                )
            raise ValueError(
                f"{path}/schema_version: no method payload migration path registered from "
                f"{envelope.schema_version!r} to {registration.payload_schema_version!r}; "
                f"available registry keys={list(self.available_keys())!r}"
            )
        return registration.payload_model.model_validate(envelope.payload)

    def resolve_execution(
        self,
        method_ref: MethodRefSpec,
        envelope: MethodPayloadEnvelope,
        *,
        worker_execution: WorkerExecutionSpec | None = None,
    ) -> ResolvedTrainingMethod[Any]:
        """Resolve and construct one method exactly once for runtime execution."""
        registration = self.resolve(method_ref, path="/method_ref")
        payload = self.validate_payload(method_ref, envelope, path="/method_payload")
        method_key = method_ref.key
        if worker_execution is not None:
            if worker_execution.method_contract.method_ref != method_key:
                raise ValueError(
                    "/worker_execution/method_contract/method_ref must match /method_ref; "
                    f"found {worker_execution.method_contract.method_ref!r}, "
                    f"expected {method_key!r}"
                )
            if worker_execution.effective_phase.method_ref != method_key:
                raise ValueError(
                    "/worker_execution/effective_phase/method_ref must match /method_ref; "
                    f"found {worker_execution.effective_phase.method_ref!r}, "
                    f"expected {method_key!r}"
                )
            if (
                worker_execution.method_contract.method_payload_schema_version
                != envelope.schema_version
            ):
                raise ValueError(
                    "/worker_execution/method_contract/method_payload_schema_version must "
                    "match /method_payload/schema_version; found "
                    f"{worker_execution.method_contract.method_payload_schema_version!r}, "
                    f"expected {envelope.schema_version!r}"
                )
        contract = registration.compile_contract(payload)
        if contract.method_ref != method_key:
            raise ValueError(
                "/worker_execution/method_contract/method_ref compiled by registry must "
                f"match /method_ref; found {contract.method_ref!r}, expected {method_key!r}"
            )
        if contract.method_payload_schema_version != envelope.schema_version:
            raise ValueError(
                "/worker_execution/method_contract/method_payload_schema_version compiled by "
                "registry must match /method_payload/schema_version; found "
                f"{contract.method_payload_schema_version!r}, expected {envelope.schema_version!r}"
            )
        program = self.program(method_key)
        if worker_execution is not None and program is not None:
            if worker_execution.method_contract != contract:
                raise ValueError(
                    "/worker_execution/method_contract must exactly match the payload-compiled "
                    f"contract for method_ref {method_key!r}"
                )
        update_kernels = registration.update_kernels_factory(payload)
        guard_predicates = registration.guard_predicates_factory(payload)
        if not isinstance(update_kernels, Mapping):
            raise TypeError(f"training method {method_key!r} returned non-mapping update kernels")
        if not isinstance(guard_predicates, Mapping):
            raise TypeError(f"training method {method_key!r} returned non-mapping guard predicates")
        from feedbax.training.worker_validation import validate_worker_contract

        effective_phase = validate_worker_contract(
            contract,
            update_kernels=update_kernels,
            guard_predicates=guard_predicates,
        )
        if (
            worker_execution is not None
            and program is not None
            and worker_execution.effective_phase != effective_phase
        ):
            raise ValueError(
                "/worker_execution/effective_phase must exactly match the validated "
                f"effective phase for method_ref {method_key!r}"
            )
        return ResolvedTrainingMethod(
            program=program,
            registration=registration,
            payload=payload,
            contract=contract,
            effective_phase=effective_phase,
            update_kernels=dict(update_kernels),
            guard_predicates=dict(guard_predicates),
        )

    def register_manifest_metadata_projection(
        self,
        registration: TrainingManifestMetadataProjectionRegistration,
    ) -> None:
        """Register projection governance independently of training methods."""
        self._require_mutable()
        required = {
            "source_payload_kind": registration.source_payload_kind,
            "source_payload_schema_id": registration.source_payload_schema_id,
            "source_payload_schema_version": registration.source_payload_schema_version,
            "projection_schema_id": registration.projection_schema_id,
            "projection_schema_version": registration.projection_schema_version,
            "owner": registration.owner,
            "package": registration.package,
        }
        missing = [name for name, value in required.items() if not value]
        if missing:
            raise ValueError(
                f"manifest metadata projection registration is incomplete: empty fields={missing!r}"
            )
        if registration.values_model.model_config.get("extra") != "forbid":
            raise ValueError("manifest metadata projection values_model must set extra='forbid'")
        if registration.values_model.model_config.get("strict") is not True:
            raise ValueError("manifest metadata projection values_model must set strict=True")
        key = registration.source_key
        if key in self._metadata_projection_registrations:
            raise ValueError(
                "manifest metadata projection already registered for source payload "
                f"identity={key!r}"
            )
        self._metadata_projection_registrations[key] = registration

    def seal(self) -> None:
        self._sealed = True

    def _require_mutable(self) -> None:
        if self._sealed:
            raise RuntimeError("training method registry is sealed")

    def resolve_manifest_metadata_projection(
        self,
        source_key: tuple[str, str, str],
        *,
        path: str,
    ) -> TrainingManifestMetadataProjectionRegistration:
        """Resolve projection governance for an exact source payload identity."""
        try:
            return self._metadata_projection_registrations[source_key]
        except KeyError as exc:
            raise ValueError(
                f"{path}: no manifest metadata projection registered for source payload "
                f"identity={source_key!r}"
            ) from exc

    def validate_manifest_metadata_projection(
        self,
        projection: TrainingManifestMetadataProjection,
        *,
        path: str,
    ) -> tuple[TrainingManifestMetadataProjectionRegistration, dict[str, Any]]:
        """Validate one projection envelope and return JSON-mode canonical values."""
        registration = self.resolve_manifest_metadata_projection(
            (
                projection.source_payload_kind,
                projection.source_payload_schema_id,
                projection.source_payload_schema_version,
            ),
            path=path,
        )
        if projection.projection_schema_id != registration.projection_schema_id:
            raise ValueError(
                f"{path}/projection_schema_id unsupported schema_id "
                f"{projection.projection_schema_id!r}; expected "
                f"{registration.projection_schema_id!r}"
            )
        if projection.projection_schema_version != registration.projection_schema_version:
            raise ValueError(
                f"{path}/projection_schema_version unsupported schema version "
                f"{projection.projection_schema_version!r}; expected "
                f"{registration.projection_schema_version!r}; "
                "migration_intentionally_absent=yes"
            )
        validated = registration.values_model.model_validate(projection.values)
        return registration, validated.model_dump(mode="json", exclude_none=True)


def standard_supervised_method_ref() -> MethodRefSpec:
    """Return the public method ref for Feedbax's standard supervised method."""
    return MethodRefSpec(package="feedbax", name="standard_supervised", version="v1")


def standard_supervised_method_payload() -> MethodPayloadEnvelope:
    """Return a default governed payload envelope for the standard supervised method."""
    return MethodPayloadEnvelope(
        schema_id=STANDARD_SUPERVISED_METHOD_PAYLOAD_SCHEMA_ID,
        schema_version=STANDARD_SUPERVISED_METHOD_PAYLOAD_SCHEMA_VERSION,
        payload=StandardSupervisedMethodPayload().model_dump(mode="json"),
    )


def standard_supervised_method_contract() -> MethodContractSpec:
    """Return the worker declaration for Feedbax's standard supervised method."""
    program = PhaseProgramSpec(
        phases=[
            PhaseSpec(
                name="train_batch",
                kind="outer_loop",
                reads=["model", "optimizer", "prng", "objective", "batch_counter"],
                writes=["model", "optimizer", "prng", "train_loss", "batch_counter"],
                update_steps=["supervised_gradient_update"],
                checkpoint_barrier="after_train_batch",
            )
        ],
        initial_phase="train_batch",
        update_steps=[
            UpdateStepSpec(
                name="supervised_gradient_update",
                kind="gradient",
                kernel=UpdateKernelSpec(
                    kernel_ref="feedbax.training.standard_supervised.gradient_update"
                ),
                reads=["model", "optimizer", "prng", "objective", "batch_counter"],
                writes=["model", "optimizer", "prng", "train_loss", "batch_counter"],
                axes=["batch"],
                optimizer_binding="optimizer_to_model",
            )
        ],
        optimizer_bindings=[
            OptimizerTargetBinding(
                name="optimizer_to_model",
                optimizer_slot="optimizer",
                target_slot="model",
                projection="after_step",
                phase_scope=["train_batch"],
                objective_reads=["objective"],
            )
        ],
        checkpoint_barriers=[
            CheckpointBarrierSpec(
                name="after_train_batch",
                phase="train_batch",
                slots=[
                    CheckpointSlotSpec(slot="model"),
                    CheckpointSlotSpec(slot="optimizer"),
                    CheckpointSlotSpec(slot="prng"),
                    CheckpointSlotSpec(slot="batch_counter"),
                ],
            )
        ],
        batch_progress=TrainingBatchProgressSpec(slot="batch_counter"),
    )
    return MethodContractSpec(
        method_ref=STANDARD_SUPERVISED_METHOD_REF,
        method_payload_schema_version=STANDARD_SUPERVISED_METHOD_PAYLOAD_SCHEMA_VERSION,
        axes=[AxisSpec(name="batch", role="batch")],
        state_slots=[
            StateSlotSpec(name="model", role="model"),
            StateSlotSpec(name="optimizer", role="optimizer"),
            StateSlotSpec(name="prng", role="prng"),
            StateSlotSpec(name="objective", role="objective"),
            StateSlotSpec(name="train_loss", role="metric", required=False),
            StateSlotSpec(name="batch_counter", role="auxiliary"),
        ],
        phase_program=program,
    )


def standard_supervised_effective_phase_spec() -> EffectivePhaseSpec:
    """Return an effective phase bundle for the standard supervised method."""
    contract = standard_supervised_method_contract()
    return EffectivePhaseSpec(
        method_ref=contract.method_ref,
        axes=contract.axes,
        state_slots=contract.state_slots,
        phase_program=contract.phase_program,
        consistency_predicate=derive_consistency_predicate(contract.phase_program),
    )


def standard_supervised_update_kernels(
    _payload: BaseModel | None = None,
) -> Mapping[str, Callable[..., Mapping[str, Any]]]:
    """Return generic kernels for the standard supervised worker declaration.

    The kernel is intentionally method-neutral: it updates declared slots and lets
    real trainers provide richer kernels by overriding the registry entry.
    """

    def gradient_update(
        slots: Mapping[str, Any],
        coordinate: ProgressCoordinate,
        context: Mapping[str, Any],
    ) -> Mapping[str, Any]:
        del context
        updates: dict[str, Any] = {
            "model": slots["model"],
            "optimizer": slots["optimizer"],
            "prng": slots["prng"],
            "train_loss": float(coordinate.program_step + 1),
            "batch_counter": slots["batch_counter"] + 1,
        }
        model = slots["model"]
        optimizer = slots["optimizer"]
        if isinstance(optimizer, Mapping) and "count" in optimizer:
            updates["optimizer"] = {**dict(optimizer), "count": optimizer["count"] + 1}
            try:
                updates["model"] = model + optimizer["count"]
            except TypeError:
                updates["model"] = model
        return updates

    return {"feedbax.training.standard_supervised.gradient_update": gradient_update}


def standard_supervised_training_program() -> DeclaredTrainingProgram[
    StandardSupervisedMethodPayload
]:
    """Return Feedbax's standard supervised training-program declaration."""
    return declare_training_program(
        method_ref=STANDARD_SUPERVISED_METHOD_REF,
        payload_schema_id=STANDARD_SUPERVISED_METHOD_PAYLOAD_SCHEMA_ID,
        payload_schema_version=STANDARD_SUPERVISED_METHOD_PAYLOAD_SCHEMA_VERSION,
        payload_model=StandardSupervisedMethodPayload,
        contract_compiler=lambda _payload: standard_supervised_method_contract(),
        update_kernels_factory=standard_supervised_update_kernels,
        schedule_projector=TrainingMethodScheduleProjector(
            projector_id="feedbax.training.standard_supervised.schedule_projection",
            projector_version="v1",
            projector=lambda _payload, _coordinates: ScheduleProjection(),
        ),
        optimizer_spec_projector=lambda payload: payload.optimizer,
        optimizer_step_extractor=_standard_supervised_optimizer_step,
        rejected_payload_versions=("feedbax.spec.training_method.standard_supervised_payload.v0",),
        owner="feedbax.contracts.training",
        package="feedbax",
    )


def _standard_supervised_optimizer_step(
    payload: StandardSupervisedMethodPayload,
    runtime: Mapping[str, Any],
) -> int:
    """Extract the standard method's declared optimizer count without tree search."""
    value: Any
    metadata = runtime.get("metadata")
    if isinstance(metadata, Mapping) and "optimizer_step" in metadata:
        value = metadata["optimizer_step"]
    elif "optimizer_step" in runtime:
        value = runtime["optimizer_step"]
    else:
        optimizer = runtime.get(payload.optimizer_slot)
        if not isinstance(optimizer, Mapping) or "count" not in optimizer:
            raise ValueError(
                "standard supervised optimizer step requires either checkpoint "
                "/metadata/optimizer_step or the declared optimizer slot count"
            )
        value = optimizer["count"]
    item = getattr(value, "item", None)
    if callable(item):
        size = getattr(value, "size", 1)
        if size != 1:
            raise ValueError("standard supervised optimizer count must contain one scalar")
        value = item()
    if isinstance(value, bool) or not isinstance(value, numbers.Real):
        raise ValueError("standard supervised optimizer count must be numeric")
    if isinstance(value, float) and not value.is_integer():
        raise ValueError("standard supervised optimizer count must be integer-valued")
    step = int(value)
    if step < 0:
        raise ValueError("standard supervised optimizer count must be non-negative")
    return step


def default_training_program_registry() -> TrainingProgramRegistry:
    """Return the default method-ref keyed payload registry."""
    registry = TrainingProgramRegistry()
    registry.register_program(standard_supervised_training_program())
    return registry


class TrainingRunSpec(TrainingRunContractModel):
    """Public durable request contract for one Feedbax training run."""

    schema_id: str = TRAINING_RUN_SPEC_SCHEMA_ID
    schema_version: str = TRAINING_RUN_SPEC_SCHEMA_VERSION
    on_nan: Literal["raise", "halt_restore_checkpoint"] = "raise"
    graph: GraphTopologySourceSpec
    task: TaskSpec
    training_config: TrainingConfig
    objective: ObjectiveSlotSpec
    risk_aggregation: RiskAggregationSpec = Field(default_factory=RiskAggregationSpec)
    method_ref: MethodRefSpec
    method_payload: MethodPayloadEnvelope
    method_extensions: MethodExtensionsSpec = Field(default_factory=MethodExtensionsSpec)
    worker_execution: WorkerExecutionSpec
    artifacts: ArtifactPolicySpec = Field(default_factory=ArtifactPolicySpec)
    checkpoint_progress: CheckpointProgressPolicySpec = Field(
        default_factory=CheckpointProgressPolicySpec
    )
    metadata: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _validate_contract(self) -> "TrainingRunSpec":
        if self.schema_id != TRAINING_RUN_SPEC_SCHEMA_ID:
            raise ValueError(
                f"/schema_id unsupported TrainingRunSpec schema_id {self.schema_id!r}; "
                f"expected {TRAINING_RUN_SPEC_SCHEMA_ID!r}"
            )
        if self.schema_version != TRAINING_RUN_SPEC_SCHEMA_VERSION:
            raise ValueError(
                "/schema_version unsupported TrainingRunSpec schema_version "
                f"{self.schema_version!r}; expected {TRAINING_RUN_SPEC_SCHEMA_VERSION!r}"
            )

        method_key = self.method_ref.key
        method_contract = self.worker_execution.method_contract
        effective_phase = self.worker_execution.effective_phase
        if method_contract.method_ref != method_key:
            raise ValueError(
                "/worker_execution/method_contract/method_ref must match /method_ref; "
                f"found {method_contract.method_ref!r}, expected {method_key!r}"
            )
        if effective_phase.method_ref != method_key:
            raise ValueError(
                "/worker_execution/effective_phase/method_ref must match /method_ref; "
                f"found {effective_phase.method_ref!r}, expected {method_key!r}"
            )
        if method_contract.method_payload_schema_version != self.method_payload.schema_version:
            raise ValueError(
                "/worker_execution/method_contract/method_payload_schema_version must match "
                f"/method_payload/schema_version; found "
                f"{method_contract.method_payload_schema_version!r}, expected "
                f"{self.method_payload.schema_version!r}"
            )
        if not method_contract.axes:
            raise ValueError("/worker_execution/method_contract/axes must declare worker axes")
        if not method_contract.state_slots:
            raise ValueError(
                "/worker_execution/method_contract/state_slots must declare worker state slots"
            )
        return self


def resolve_training_run_spec(
    spec: TrainingRunSpec, registry: TrainingProgramRegistry
) -> ResolvedTrainingMethod[Any]:
    """Resolve registry-owned training semantics after structural parsing."""
    return registry.resolve_execution(
        spec.method_ref,
        spec.method_payload,
        worker_execution=spec.worker_execution,
    )


def validate_training_run_spec_semantics(
    spec: TrainingRunSpec, registry: TrainingProgramRegistry
) -> TrainingRunSpec:
    """Validate registry-owned semantics at an explicit post-bootstrap boundary."""
    resolve_training_run_spec(spec, registry)
    return spec


# Enable forward references
LossTermSpec.model_rebuild()
