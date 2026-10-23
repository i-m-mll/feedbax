"""Pydantic models for component definitions."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from string import Formatter
from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field, model_validator

from feedbax.contracts.domain import CAUSAL_DOMAIN_ID
from feedbax.contracts.acausal import AcausalGraphSpec
from feedbax.contracts.graph import GraphSpec, GraphUIState, ParamSchema, ParamValue
from feedbax.contracts.representation import RepresentationSpec


COMPONENT_DEFINITION_SCHEMA_ID = "feedbax.spec.component_definition"
COMPONENT_DEFINITION_SCHEMA_VERSION_V1 = "feedbax.spec.component_definition.v1"
COMPONENT_DEFINITION_SCHEMA_VERSION_V2 = "feedbax.spec.component_definition.v2"
COMPONENT_DEFINITION_SCHEMA_VERSION = "feedbax.spec.component_definition.v3"
COMPONENT_DEFINITION_PORT_KIND_MIGRATION_ID = "component-definition-v1-to-v2-port-kind"
COMPONENT_DEFINITION_DYNAMIC_PORT_POLICY_MIGRATION_ID = (
    "component-definition-v2-to-v3-dynamic-port-policy"
)


def _migrate_port_type_payload(payload: Any) -> Any:
    if not isinstance(payload, dict):
        return payload
    if "kind" in payload:
        return payload
    return {**payload, "kind": "signal"}


def migrate_component_definition_v1_to_v2_payload(payload: Dict[str, Any]) -> Dict[str, Any]:
    """Migrate legacy component definitions to explicit port kinds."""

    schema_version = payload.get("schema_version")
    if schema_version not in {None, COMPONENT_DEFINITION_SCHEMA_VERSION_V1}:
        return dict(payload)
    migrated = dict(payload)
    migrated.setdefault("schema_id", COMPONENT_DEFINITION_SCHEMA_ID)
    migrated["schema_version"] = COMPONENT_DEFINITION_SCHEMA_VERSION_V2
    port_types = migrated.get("port_types")
    if isinstance(port_types, dict):
        migrated["port_types"] = {
            **port_types,
            "inputs": {
                key: _migrate_port_type_payload(value)
                for key, value in dict(port_types.get("inputs") or {}).items()
            },
            "outputs": {
                key: _migrate_port_type_payload(value)
                for key, value in dict(port_types.get("outputs") or {}).items()
            },
        }
    return migrated


def migrate_component_definition_v2_to_v3_payload(payload: Dict[str, Any]) -> Dict[str, Any]:
    """Add the optional declarative dynamic-port policy field."""

    if payload.get("schema_version") != COMPONENT_DEFINITION_SCHEMA_VERSION_V2:
        return dict(payload)
    migrated = dict(payload)
    migrated["schema_version"] = COMPONENT_DEFINITION_SCHEMA_VERSION
    migrated.setdefault("dynamic_port_policy", None)
    return migrated


def migrate_component_definition_payload(payload: Dict[str, Any]) -> Dict[str, Any]:
    """Migrate legacy component definitions to the current schema."""

    migrated = migrate_component_definition_v1_to_v2_payload(payload)
    return migrate_component_definition_v2_to_v3_payload(migrated)


class PortType(BaseModel):
    """Type information for a port."""

    model_config = ConfigDict(extra="forbid")

    dtype: str = "scalar"
    shape: Optional[List[int]] = None
    rank: Optional[int] = None
    kind: Literal["signal", "conserving"] = "signal"
    physical_domain: Optional[str] = None
    across_vars: Optional[List[str]] = None
    through_var: Optional[str] = None

    @model_validator(mode="before")
    @classmethod
    def migrate_legacy_signal_port(cls, data: Any) -> Any:
        return _migrate_port_type_payload(data)


class PortTypeSpec(BaseModel):
    """Port type specifications for a component."""

    model_config = ConfigDict(extra="forbid")

    inputs: Dict[str, PortType] = Field(default_factory=dict)
    outputs: Dict[str, PortType] = Field(default_factory=dict)


class DynamicPortPolicyError(ValueError):
    """Raised when a dynamic-port policy cannot produce an unambiguous layout."""


class DynamicPortPolicy(BaseModel):
    """Declarative policy for ports whose arity is derived from parameters."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    count_param: str = Field(min_length=1)
    count_mode: Literal["integer", "sequence_length"]
    direction: Literal["input", "output"]
    fixed_input_ports: tuple[str, ...] = ()
    fixed_output_ports: tuple[str, ...] = ()
    generated_name_template: str = Field(min_length=1)
    generated_index_origin: int = 0
    minimum_count: int = Field(default=1, ge=0)
    dynamic_port_type: PortType

    @model_validator(mode="after")
    def validate_port_namespace(self) -> "DynamicPortPolicy":
        _validate_fixed_port_names(self.fixed_input_ports, direction="input")
        _validate_fixed_port_names(self.fixed_output_ports, direction="output")
        template = self.generated_name_template
        non_tokens = template.replace("{index}", "")
        if "{" in non_tokens or "}" in non_tokens:
            raise ValueError(_INVALID_GENERATED_NAME_TEMPLATE)
        try:
            parsed = tuple(Formatter().parse(template))
        except (TypeError, ValueError) as exc:
            raise ValueError(_INVALID_GENERATED_NAME_TEMPLATE) from exc
        fields = [
            (field_name, format_spec, conversion)
            for _literal, field_name, format_spec, conversion in parsed
            if field_name is not None
        ]
        if not fields or any(
            field_name != "index" or format_spec != "" or conversion is not None
            for field_name, format_spec, conversion in fields
        ):
            raise ValueError(_INVALID_GENERATED_NAME_TEMPLATE)
        return self


class DynamicPortLayout(BaseModel):
    """Derived input and output port names for one component instance."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    input_ports: tuple[str, ...]
    output_ports: tuple[str, ...]


_INVALID_GENERATED_NAME_TEMPLATE = (
    "generated_name_template must contain only unformatted '{index}' replacement fields"
)


def _validate_fixed_port_names(names: Sequence[str], *, direction: str) -> None:
    if any(not name for name in names):
        raise ValueError(f"fixed {direction} port names must be non-empty")
    duplicates = sorted({name for name in names if names.count(name) > 1})
    if duplicates:
        raise ValueError(f"fixed {direction} port names must be unique: {duplicates!r}")


def _format_generated_port_name(policy: DynamicPortPolicy, index: int) -> str:
    try:
        return policy.generated_name_template.format(index=index)
    except Exception as exc:
        raise DynamicPortPolicyError(_INVALID_GENERATED_NAME_TEMPLATE) from exc


def derive_dynamic_port_count(
    policy: DynamicPortPolicy,
    params: Mapping[str, Any],
) -> int:
    """Derive and validate a dynamic port count without component-type branching."""

    if policy.count_param not in params:
        raise DynamicPortPolicyError(f"missing dynamic-port parameter {policy.count_param!r}")
    value = params[policy.count_param]
    if policy.count_mode == "integer":
        if isinstance(value, bool) or not isinstance(value, int):
            raise DynamicPortPolicyError(
                f"dynamic-port parameter {policy.count_param!r} must be an integer"
            )
        count = value
    else:
        if not isinstance(value, (list, tuple)):
            raise DynamicPortPolicyError(
                f"dynamic-port parameter {policy.count_param!r} must be a sequence"
            )
        count = len(value)
    if count < policy.minimum_count:
        raise DynamicPortPolicyError(
            f"dynamic-port parameter {policy.count_param!r} derives {count} ports; "
            f"minimum is {policy.minimum_count}"
        )
    return count


def derive_dynamic_port_layout(
    policy: DynamicPortPolicy,
    params: Mapping[str, Any],
) -> DynamicPortLayout:
    """Derive the complete port layout declared by ``policy``."""

    count = derive_dynamic_port_count(policy, params)
    generated = tuple(
        _format_generated_port_name(policy, policy.generated_index_origin + offset)
        for offset in range(count)
    )
    if any(not name for name in generated):
        raise DynamicPortPolicyError("generated dynamic port names must be non-empty")
    duplicate_generated = sorted({name for name in generated if generated.count(name) > 1})
    if duplicate_generated:
        raise DynamicPortPolicyError(
            f"generated dynamic port names must be unique: {duplicate_generated!r}"
        )
    fixed_names = (
        policy.fixed_input_ports if policy.direction == "input" else policy.fixed_output_ports
    )
    collisions = sorted(set(generated).intersection(fixed_names))
    if collisions:
        raise DynamicPortPolicyError(
            f"generated dynamic port names collide with fixed {policy.direction} ports: "
            f"{collisions!r}"
        )
    input_ports = policy.fixed_input_ports
    output_ports = policy.fixed_output_ports
    if policy.direction == "input":
        input_ports = (*input_ports, *generated)
    else:
        output_ports = (*output_ports, *generated)
    return DynamicPortLayout(input_ports=input_ports, output_ports=output_ports)


def validate_dynamic_port_layout(
    policy: DynamicPortPolicy,
    params: Mapping[str, Any],
    *,
    input_ports: Sequence[str],
    output_ports: Sequence[str],
) -> DynamicPortLayout:
    """Return the expected layout or fail when declared ports do not match it."""

    expected = derive_dynamic_port_layout(policy, params)
    declared_inputs = tuple(input_ports)
    declared_outputs = tuple(output_ports)
    if declared_inputs != expected.input_ports or declared_outputs != expected.output_ports:
        raise DynamicPortPolicyError(
            "dynamic port layout mismatch: "
            f"declared inputs={declared_inputs!r}, outputs={declared_outputs!r}; "
            f"expected inputs={expected.input_ports!r}, outputs={expected.output_ports!r}"
        )
    return expected


class ComponentIdentity(BaseModel):
    """Stable ownership and provenance for a component type."""

    type_id: str
    owner: Optional[str] = None
    provenance: Optional[str] = None
    provenance_kind: Literal["feedbax", "package", "file", "local", "unknown"] = "unknown"
    package: Optional[str] = None
    import_path: Optional[str] = None
    stable: bool = False


class ComponentMigrationInfo(BaseModel):
    """Discoverable component ID or parameter-schema migration edge."""

    migration_id: str
    owner: str
    source_type: str
    target_type: str
    source_param_schema_version: Optional[str] = None
    target_param_schema_version: Optional[str] = None
    description: str = ""


class ComponentDefinition(BaseModel):
    """Definition of a component type available in the library."""

    model_config = ConfigDict(extra="forbid")

    schema_id: Literal[COMPONENT_DEFINITION_SCHEMA_ID] = COMPONENT_DEFINITION_SCHEMA_ID
    schema_version: Literal[COMPONENT_DEFINITION_SCHEMA_VERSION] = (
        COMPONENT_DEFINITION_SCHEMA_VERSION
    )
    name: str
    category: str
    description: str
    param_schema: List[ParamSchema] = Field(default_factory=list)
    input_ports: List[str] = Field(default_factory=list)
    output_ports: List[str] = Field(default_factory=list)
    icon: str = "box"
    default_params: Dict[str, ParamValue] = Field(default_factory=dict)
    port_types: Optional[PortTypeSpec] = None
    dynamic_port_policy: Optional[DynamicPortPolicy] = None
    domain: str = CAUSAL_DOMAIN_ID
    interior_domain: Optional[str] = None
    is_composite: bool = False
    template_graph: Optional[GraphSpec | AcausalGraphSpec] = None
    template_ui_state: Optional[GraphUIState] = None
    template_id: Optional[str] = None
    template_kind: Optional[str] = None
    provenance: Optional[str] = None
    identity: Optional[ComponentIdentity] = None
    owner: Optional[str] = None
    param_schema_version: str = "1"
    supported_param_schema_versions: List[str] = Field(default_factory=lambda: ["1"])
    migrations: List[ComponentMigrationInfo] = Field(default_factory=list)
    trainable_by_default: bool = False
    representation: Optional[RepresentationSpec] = None

    @model_validator(mode="before")
    @classmethod
    def migrate_legacy_definition(cls, data: Any) -> Any:
        if not isinstance(data, dict):
            return data
        return migrate_component_definition_payload(data)
