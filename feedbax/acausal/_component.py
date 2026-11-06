"""Derive catalogue options from an acausal element's physical ports."""

from __future__ import annotations

import inspect
from typing import Annotated

from pydantic import Field

from feedbax.acausal.base import AcausalElement
from feedbax.contracts.acausal_interface import conserving_port_type
from feedbax.contracts.component import PortTypeSpec

NonNegativeFloat = Annotated[float, Field(ge=0)]


def element_options(element_type: type[AcausalElement]) -> dict:
    ports = element_type("__catalogue__").ports
    return {
        "description": inspect.getdoc(element_type) or "",
        "build_context_fields": ("name",),
        "input_ports": tuple(sorted(ports)),
        "output_ports": (),
        "port_types": PortTypeSpec(
            inputs={
                name: conserving_port_type(
                    physical_domain=port.domain.value,
                    across_vars=port.across_vars,
                    through_var=port.through_var,
                )
                for name, port in ports.items()
            },
        ),
    }
