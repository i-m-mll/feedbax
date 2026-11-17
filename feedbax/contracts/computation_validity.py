"""Portable evidence from the numerical boundary of a managed computation."""

from __future__ import annotations

from typing import Annotated, Literal

from pydantic import Field, FiniteFloat, NonNegativeInt, PositiveInt, model_validator

from feedbax.contracts.artifact_custody import ContentObjectRef, ContentTreeRef
from feedbax.contracts.request_refs import Name, ReferenceValue, Sha256


NORMALIZATION_VERSION = 1
REFERENCE_PLATFORM = "cpu"


class DriverIdentity(ReferenceValue):
    name: Name
    revision: PositiveInt


class ArraySignature(ReferenceValue):
    shape: tuple[NonNegativeInt, ...]
    dtype: Name
    weak_type: bool
    placement: Sha256


class ArgumentSignature(ReferenceValue):
    arrays: tuple[ArraySignature, ...]
    static_sha256: Sha256


class ArgumentIdentity(ReferenceValue):
    signature: ArgumentSignature
    sha256: Sha256


class ComputationStart(ReferenceValue):
    slot: Name
    arguments: ArgumentIdentity
    binding: ContentObjectRef | None = None
    binding_inputs: tuple[ContentObjectRef | ContentTreeRef, ...] = ()


class EnvironmentCall(ReferenceValue):
    kind: Literal["environment"] = "environment"
    target: Name


class OutputFreeCall(ReferenceValue):
    kind: Literal["output_free"] = "output_free"
    target: Name


class AdapterCall(ReferenceValue):
    kind: Literal["adapter"] = "adapter"
    target: Name
    adapter: Name
    distributions: dict[Name, Name] = Field(min_length=1)
    implementation_sha256: Sha256
    static_sha256: Sha256


class UnidentifiedCall(ReferenceValue):
    kind: Literal["unidentified"] = "unidentified"
    target: Name


type ForeignCall = Annotated[
    EnvironmentCall | OutputFreeCall | AdapterCall | UnidentifiedCall,
    Field(discriminator="kind"),
]


class LoweredProgram(ReferenceValue):
    platform: Name
    sha256: Sha256
    calls: tuple[ForeignCall, ...]
    raw_program: ContentObjectRef | None = None


class ProgramIdentity(ReferenceValue):
    slot: Name
    specialization: ArgumentSignature
    reference: LoweredProgram | None
    execution: LoweredProgram

    @model_validator(mode="after")
    def _reference_target(self) -> ProgramIdentity:
        if self.reference is not None and self.reference.platform != REFERENCE_PLATFORM:
            raise ValueError("The lowered reference program must target CPU")
        if self.execution.platform == REFERENCE_PLATFORM and self.reference != self.execution:
            raise ValueError("A CPU execution is its own reference program")
        return self


class TrainPrefix(ReferenceValue):
    kind: Literal["train"] = "train"
    precision: Literal["float32", "float64"]
    losses: tuple[FiniteFloat, ...] = Field(min_length=1, max_length=5)
    parameters: ContentObjectRef
    parameters_descriptor_sha256: Sha256


class EvaluateOutput(ReferenceValue):
    slot: Name
    values: ContentObjectRef
    descriptor_sha256: Sha256


class EvaluatePrefix(ReferenceValue):
    kind: Literal["evaluate"] = "evaluate"
    outputs: tuple[EvaluateOutput, ...] = Field(min_length=1, max_length=5)


type ComputationPrefix = Annotated[TrainPrefix | EvaluatePrefix, Field(discriminator="kind")]


class ComputationTrace(ReferenceValue):
    normalization_version: PositiveInt = NORMALIZATION_VERSION
    reference_platform: Literal["cpu"] = REFERENCE_PLATFORM
    driver: DriverIdentity
    starts: tuple[ComputationStart, ...] = Field(min_length=1)
    programs: tuple[ProgramIdentity, ...] = Field(min_length=1)
    completed_steps: PositiveInt
    prefix: ComputationPrefix | None = None

    @model_validator(mode="after")
    def _coverage(self) -> ComputationTrace:
        slots = [start.slot for start in self.starts]
        signatures = [(program.slot, program.specialization) for program in self.programs]
        if len(set(slots)) != len(slots) or set(slots) != {slot for slot, _ in signatures}:
            raise ValueError(
                "Each computation slot must have one exact start and at least one program"
            )
        if len(set(signatures)) != len(signatures):
            raise ValueError("Each computation specialization must occur once per slot")
        if isinstance(self.prefix, TrainPrefix) and self.completed_steps < len(self.prefix.losses):
            raise ValueError("The training prefix must consist of completed updates")
        if isinstance(self.prefix, EvaluatePrefix) and (
            [output.slot for output in self.prefix.outputs] != slots[: len(self.prefix.outputs)]
            or self.completed_steps < len(self.prefix.outputs)
        ):
            raise ValueError("The evaluation prefix must name the first completed slots")
        return self

    def content_references(self) -> tuple[ContentObjectRef | ContentTreeRef, ...]:
        """Return the custody edges needed to read and repeat the computation."""
        refs = [reference for start in self.starts for reference in start.binding_inputs]
        refs.extend(start.binding for start in self.starts if start.binding is not None)
        refs.extend(
            lowering.raw_program
            for program in self.programs
            for lowering in (program.reference, program.execution)
            if lowering is not None and lowering.raw_program is not None
        )
        if isinstance(self.prefix, TrainPrefix):
            refs.append(self.prefix.parameters)
        elif isinstance(self.prefix, EvaluatePrefix):
            refs.extend(output.values for output in self.prefix.outputs)
        return tuple(
            {
                (
                    (ref.tree_id, ref.descriptor.size_bytes)
                    if isinstance(ref, ContentTreeRef)
                    else (ref.object_id, ref.size_bytes)
                ): ref
                for ref in refs
            }.values()
        )


class DriverChanged(ReferenceValue):
    kind: Literal["driver"] = "driver"
    previous: DriverIdentity
    present: DriverIdentity


class StartsChanged(ReferenceValue):
    kind: Literal["arguments"] = "arguments"
    slots: tuple[Name, ...]


class ProgramsChanged(ReferenceValue):
    kind: Literal["program"] = "program"
    slots: tuple[Name, ...]


class StepCountChanged(ReferenceValue):
    kind: Literal["steps"] = "steps"
    previous: PositiveInt
    present: PositiveInt


type ComputationChange = Annotated[
    DriverChanged | StartsChanged | ProgramsChanged | StepCountChanged,
    Field(discriminator="kind"),
]


class ComputationEqual(ReferenceValue):
    status: Literal["equal"] = "equal"


class ComputationDifferent(ReferenceValue):
    status: Literal["different"] = "different"
    changes: tuple[ComputationChange, ...] = Field(min_length=1)


class ComputationUnconfirmable(ReferenceValue):
    status: Literal["unconfirmable"] = "unconfirmable"
    reason: Literal["normalization", "reference_unavailable", "execution_target", "foreign_call"]
    targets: tuple[Name, ...] = ()


type ComputationComparison = Annotated[
    ComputationEqual | ComputationDifferent | ComputationUnconfirmable,
    Field(discriminator="status"),
]


class PrefixDrift(ReferenceValue):
    max_relative_loss: FiniteFloat | None
    relative_parameter_l2: FiniteFloat | None


class PrefixConfirmed(ReferenceValue):
    status: Literal["confirmed"] = "confirmed"
    drift: PrefixDrift


class PrefixChanged(ReferenceValue):
    status: Literal["different"] = "different"
    reason: Literal["arguments", "output_shape_or_dtype", "numerical_drift", "nonfinite_outputs"]
    drift: PrefixDrift | None = None


class PrefixUnavailable(ReferenceValue):
    status: Literal["unavailable"] = "unavailable"
    reason: Literal["profile", "output_unavailable", "unreadable_witness"]


type PrefixComparison = Annotated[
    PrefixConfirmed | PrefixChanged | PrefixUnavailable,
    Field(discriminator="status"),
]
