"""Per-layer authored-document budgets: the Feedbax-owned document, project values.

Closed vocabulary bounds *what* an authored envelope can say; budgets bound *how
much*. Both are needed before "the authored artifact carries only the delta" is
true by construction rather than by convention.

The split of ownership is exact and deliberate:

* **Feedbax owns the document schema and the enforcement.** The cap names, their
  meaning, the layer-sectioned shape, and the walk that refuses an over-budget
  document are engine mechanism and live here.
* **The project owns the numbers.** A project states one section per layer of
  the :mod:`~feedbax.contracts.experiment_envelope_dialect`, and the sections are
  checked against exactly that set. Feedbax never ships a default budget and
  never infers a cap; the project cannot omit a layer or invent one, because the
  layers are the dialect's and not the project's.

A budget is per layer because layers differ in natural size: a parameter delta
and a block of prose are not over-budget at the same number. Enforcement reads
the section belonging to the layer the document authors, and the *widest* cap
any section states is what a document is judged under before its layer is known.

Caps Feedbax enforces are a closed set of eight. A project cap the engine does
not enforce is not silently ignored and is not a schema error either: it goes in
a section's ``project_caps`` block, where it is validated as a positive integer
and handed back to the project's own compiler to enforce. That keeps a
project-specific bound expressible without teaching the engine what it means.
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, NoReturn

from feedbax.contracts.experiment_envelope import (
    ExperimentEnvelopeRejection,
    ExperimentEnvelopeRejectionCategory,
)

AUTHORING_BUDGET_SCHEMA_ID = "feedbax.spec.authoring_budget"
AUTHORING_BUDGET_SCHEMA_VERSION_V1 = f"{AUTHORING_BUDGET_SCHEMA_ID}.v1"
AUTHORING_BUDGET_SCHEMA_VERSION = AUTHORING_BUDGET_SCHEMA_VERSION_V1

#: The only budget-policy versions read. Enumerated, never inferred.
AUTHORING_BUDGET_SUPPORTED_SCHEMA_VERSIONS: tuple[str, ...] = (
    AUTHORING_BUDGET_SCHEMA_VERSION_V1,
)

#: Versions this loader accepts, mapped to the version they migrate to. Empty at
#: v1: there is no older Feedbax-owned budget document. A future version adds its
#: edge here and to ``default_spec_registry`` in one change.
AUTHORING_BUDGET_MIGRATION_TABLE: dict[str, str] = {}

#: The label a document is judged under before its layer is known: the widest cap
#: any section states, per dimension. It is not a layer and admits nothing the
#: closed authored schema would otherwise refuse.
WIDEST_LAYER_LABEL = "widest"


@dataclass(frozen=True)
class LayerBudget:
    """One layer's authored-document caps.

    ``layer`` is carried so every refusal can say *whose* budget was exceeded.
    ``project_caps`` holds bounds the engine validates but does not enforce; the
    project's own compiler reads them.

    Attributes:
        layer: The declared layer label these caps belong to.
        max_lines: Cap on canonically formatted lines of the whole document.
        max_bytes: Cap on the raw authored byte length.
        max_scalar_bytes: Cap on the UTF-8 length of any single string scalar.
        max_ast_nodes: Cap on total parsed syntax nodes.
        max_depth: Cap on nesting depth.
        max_items: Cap on the length of any single array.
        max_object_keys: Cap on the key count of any single object.
        max_assertions: Cap on authored assertions.
        project_caps: Positive-integer bounds the project enforces itself.
    """

    layer: str
    max_lines: int
    max_bytes: int
    max_scalar_bytes: int
    max_ast_nodes: int
    max_depth: int
    max_items: int
    max_object_keys: int
    max_assertions: int
    project_caps: Mapping[str, int] = MappingProxyType({})

    def cap(self, name: str) -> str:
        """Name one of this layer's caps for a refusal message."""
        return f"the {self.layer} layer's authored budget of {getattr(self, name)}"

    def project_cap(self, name: str) -> int:
        """Return one project-enforced cap, or fail closed naming what is stated."""
        try:
            return self.project_caps[name]
        except KeyError as exc:
            stated = ", ".join(sorted(self.project_caps)) or "none"
            raise KeyError(
                f"the {self.layer!r} layer states no project cap {name!r}; stated: {stated}"
            ) from exc


#: The eight caps the engine enforces, derived from the dataclass rather than
#: re-spelled, so a new enforced cap is added in exactly one place.
ENFORCED_BUDGET_KEYS: tuple[str, ...] = tuple(
    name
    for name in LayerBudget.__dataclass_fields__
    if name not in ("layer", "project_caps")
)


def _reject(
    category: ExperimentEnvelopeRejectionCategory, field: str, message: str
) -> NoReturn:
    raise ExperimentEnvelopeRejection(category, message, field=field)


def _require_positive_int(value: Any, field: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        _reject(
            ExperimentEnvelopeRejectionCategory.INVALID_VALUE,
            field,
            f"a cap is a positive integer; found {value!r}",
        )
    return int(value)


@dataclass(frozen=True)
class AuthoringBudgets:
    """One project's per-layer authored-document budget policy."""

    budget_id: str
    layers: Mapping[str, LayerBudget]

    @classmethod
    def from_document(
        cls,
        document: Any,
        *,
        field: str,
        declared_layers: Iterable[str],
    ) -> "AuthoringBudgets":
        """Build the policy from a document already known to declare a read version.

        The sections are validated exhaustively against the project's declared
        layer labels: exactly those layers, exactly the eight enforced caps in
        each, every value a positive integer. A typo in a cap name is a refusal
        here rather than a silently absent bound.
        """
        expected = set(declared_layers)
        if not expected:
            _reject(
                ExperimentEnvelopeRejectionCategory.INVALID_VALUE,
                field,
                "a project that declares no layer has no budget to state",
            )
        sections = document.get("layers")
        if not isinstance(sections, dict) or set(sections) != expected:
            found = sorted(sections) if isinstance(sections, dict) else sections
            _reject(
                ExperimentEnvelopeRejectionCategory.INVALID_VALUE,
                f"{field}#layers",
                "the budget states one section per declared layer; expected exactly "
                f"{sorted(expected)}, found {found!r}",
            )
        budget_id = document.get("budget_id")
        if not isinstance(budget_id, str) or not budget_id.strip():
            _reject(
                ExperimentEnvelopeRejectionCategory.MISSING_FIELD,
                f"{field}#budget_id",
                "a budget document names the resource identity it supplies",
            )
        layers: dict[str, LayerBudget] = {}
        for layer in sorted(expected):
            section = sections[layer]
            locator = f"{field}#layers.{layer}"
            if not isinstance(section, dict):
                _reject(
                    ExperimentEnvelopeRejectionCategory.INVALID_VALUE,
                    locator,
                    f"a layer section is an object; found {section!r}",
                )
            enforced = {key: value for key, value in section.items() if key != "project_caps"}
            if set(enforced) != set(ENFORCED_BUDGET_KEYS):
                _reject(
                    ExperimentEnvelopeRejectionCategory.INVALID_VALUE,
                    locator,
                    f"expected exactly the caps {sorted(ENFORCED_BUDGET_KEYS)}, found "
                    f"{sorted(enforced)}",
                )
            caps = {
                name: _require_positive_int(value, f"{locator}.{name}")
                for name, value in enforced.items()
            }
            project_caps = section.get("project_caps", {})
            if not isinstance(project_caps, dict):
                _reject(
                    ExperimentEnvelopeRejectionCategory.INVALID_VALUE,
                    f"{locator}.project_caps",
                    f"project caps are an object of positive integers; found {project_caps!r}",
                )
            layers[layer] = LayerBudget(
                layer=layer,
                project_caps=MappingProxyType(
                    {
                        name: _require_positive_int(value, f"{locator}.project_caps.{name}")
                        for name, value in sorted(project_caps.items())
                    }
                ),
                **caps,
            )
        return cls(budget_id=str(budget_id), layers=MappingProxyType(layers))

    def for_layer(self, layer: str) -> LayerBudget:
        """Return the caps for one declared layer, or fail closed."""
        try:
            return self.layers[layer]
        except KeyError as exc:
            known = ", ".join(sorted(self.layers)) or "none"
            raise KeyError(
                f"budget {self.budget_id!r} states no section for layer {layer!r}; "
                f"sections: {known}"
            ) from exc

    @property
    def widest(self) -> LayerBudget:
        """Return the widest cap any layer states, per dimension.

        This is what the pre-parse guard and a document whose layer cannot yet be
        determined are judged under. It admits nothing the closed authored schema
        would otherwise refuse; it only decides which diagnostic a document gets.
        """
        return LayerBudget(
            layer=WIDEST_LAYER_LABEL,
            **{
                name: max(getattr(caps, name) for caps in self.layers.values())
                for name in ENFORCED_BUDGET_KEYS
            },
        )


def load_authoring_budget_document(raw: bytes, *, field: str) -> dict[str, Any]:
    """Parse one budget document and fail closed on an unread schema version.

    Supported versions are enumerated. A version absent from both the supported
    set and the migration table has no path forward and is refused with both
    named; it is never coerced, guessed at, or read through a compatibility
    fallback.
    """
    try:
        document = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ExperimentEnvelopeRejection(
            ExperimentEnvelopeRejectionCategory.NONCANONICAL_FORMAT,
            f"the authoring budget at {field} is not readable JSON: {exc}",
            field=field,
        ) from exc
    if not isinstance(document, dict):
        _reject(
            ExperimentEnvelopeRejectionCategory.INVALID_VALUE,
            field,
            "an authoring budget document is a JSON object",
        )
    found_id = document.get("schema_id")
    if found_id != AUTHORING_BUDGET_SCHEMA_ID:
        _reject(
            ExperimentEnvelopeRejectionCategory.UNSUPPORTED_SCHEMA_VERSION,
            f"{field}#schema_id",
            f"expected schema_id {AUTHORING_BUDGET_SCHEMA_ID!r}, found {found_id!r}",
        )
    version = document.get("schema_version")
    if version not in AUTHORING_BUDGET_SUPPORTED_SCHEMA_VERSIONS:
        _reject(
            ExperimentEnvelopeRejectionCategory.UNSUPPORTED_SCHEMA_VERSION,
            f"{field}#schema_version",
            f"unsupported schema_version {version!r}; "
            f"supported={list(AUTHORING_BUDGET_SUPPORTED_SCHEMA_VERSIONS)}; "
            f"migration table={AUTHORING_BUDGET_MIGRATION_TABLE!r}; "
            "migration_intentionally_absent=yes",
        )
    return document


__all__ = [
    "AUTHORING_BUDGET_MIGRATION_TABLE",
    "AUTHORING_BUDGET_SCHEMA_ID",
    "AUTHORING_BUDGET_SCHEMA_VERSION",
    "AUTHORING_BUDGET_SCHEMA_VERSION_V1",
    "AUTHORING_BUDGET_SUPPORTED_SCHEMA_VERSIONS",
    "ENFORCED_BUDGET_KEYS",
    "WIDEST_LAYER_LABEL",
    "AuthoringBudgets",
    "LayerBudget",
    "load_authoring_budget_document",
]
