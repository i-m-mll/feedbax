"""Portable evidence from the numerical boundary of a managed computation."""

from __future__ import annotations

from typing import Annotated, Literal

from pydantic import Field, FiniteFloat, NonNegativeInt, PositiveInt, model_validator

from feedbax.contracts.artifact_custody import ContentObjectRef
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
    losses: tuple[FiniteFloat, FiniteFloat, FiniteFloat, FiniteFloat, FiniteFloat]
    parameters: ContentObjectRef


class ComputationTrace(ReferenceValue):
    normalization_version: PositiveInt = NORMALIZATION_VERSION
    reference_platform: Literal["cpu"] = REFERENCE_PLATFORM
    driver: DriverIdentity
    starts: tuple[ComputationStart, ...] = Field(min_length=1)
    programs: tuple[ProgramIdentity, ...] = Field(min_length=1)
    completed_steps: PositiveInt
    prefix: TrainPrefix | None = None

    @model_validator(mode="after")
    def _coverage(self) -> ComputationTrace:
        slots = [start.slot for start in self.starts]
        signatures = [(program.slot, program.specialization) for program in self.programs]
        if len(set(slots)) != len(slots) or set(slots) != {slot for slot, _ in signatures}:
            raise ValueError("Each computation slot must have one exact start and at least one program")
        if len(set(signatures)) != len(signatures):
            raise ValueError("Each computation specialization must occur once per slot")
        if self.prefix is not None and self.completed_steps < len(self.prefix.losses):
            raise ValueError("The training prefix must consist of completed updates")
        return self


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
    reason: Literal["profile", "output_unavailable", "unreadable_witness", "foreign_call"]


type PrefixComparison = Annotated[
    PrefixConfirmed | PrefixChanged | PrefixUnavailable,
    Field(discriminator="status"),
]
