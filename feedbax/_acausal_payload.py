"""Ordered source payloads shared by acausal elements and graph authoring."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from feedbax.contracts.component import PortType


@dataclass(frozen=True)
class SourcePayload:
    """Ordered scalar fields in a causal payload delivered to one acausal source."""

    slots: tuple[str, ...]
    vector: bool = False
    width: int | None = None

    @property
    def shape(self) -> tuple[int, ...]:
        if self.width is not None:
            return (self.width,)
        return () if len(self.slots) == 1 and not self.vector else (len(self.slots),)

    def validate_shape(self, port: str, shape: tuple[int, ...] | None) -> None:
        expected_shape = self.shape
        # Scalar channels also accept the one-element arrays emitted by vector components.
        if shape == expected_shape or (expected_shape == () and shape == (1,)):
            return
        expected = (
            f"shape {expected_shape} with fields {self.slots}"
            if expected_shape
            else "a scalar or a one-element vector"
        )
        raise ValueError(f"Acausal source input {port!r} expects {expected}, got shape {shape}")

    def port_type(self) -> PortType:
        from feedbax.contracts.acausal_interface import signal_port_type
        from feedbax.contracts.component import PortType

        return (
            PortType(dtype="vector", shape=list(self.shape), kind="signal")
            if self.shape
            else signal_port_type()
        )


FORCE_PAYLOAD = SourcePayload(("force",))
TORQUE_PAYLOAD = SourcePayload(("torque",))
PRESCRIBED_MOTION_PAYLOAD = SourcePayload(("pos", "vel"))
ACTIVATION_PAYLOAD = SourcePayload(("activation",))
MUSCLE_EXCITATION_PAYLOAD = SourcePayload(
    (
        "shoulder_flexor",
        "shoulder_extensor",
        "elbow_flexor",
        "elbow_extensor",
        "biarticular_flexor",
        "biarticular_extensor",
    )
)
