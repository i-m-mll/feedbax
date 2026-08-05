"""Figure input role references and row expansion over a resolved row set.

An authored figure variant used to carry, per row, an echoed patch block plus
an embedded ``authenticated_manifest`` authority block naming a manifest
sha256 and byte size. Those are realization facts: they are produced by a run,
not decided by an author, and putting them in an authored document both bloats
it and invalidates every pin whenever custody moves.

This module replaces that authored form with a role reference. A figure input
names a role, and a role is either

* per-row — the role is resolved once per row in the expanded row set, from
  the row index's post-run custody bindings; or
* shared — the role is experiment-invariant and names one manifest directly.

Nothing else is admitted, and an authored reference that carries an authority
block (``manifest_sha256``, ``size_bytes``, or a nested authenticated manifest
ref) is rejected with the field named.

The custody boundary is the same one the row index draws.
:func:`resolve_figure_input_roles` is the compile-time half: it expands roles
over the resolved rows and reports which per-row roles are still awaiting a run
receipt, so a first-time figure compiles with no production record at all.
:func:`expand_figure_rows` is the bound half: given fully bound inputs it
derives the ordinary current :class:`~feedbax.contracts.figures.FigureSpec`
that the figure layer executes. Digests and byte sizes stay in the compile
lock; they never enter the compiled figure's scientific identity.

Derivation is normative and unauthored. Row-index order alone drives:

* the ``row_{n}__`` namespace applied to trace-family names, trace names,
  panel names, and per-trace data items;
* panel grid placement, one base panel block per expanded row;
* panel titles, ``"{row_label} — {base_panel_title}"``;
* legend ownership — the first expanded row keeps the base legend member and
  every later row drops it and draws with ``showlegend`` false;
* colorbar family and placement, always the first expanded row;
* assembler height, the base height times the expanded row count.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from copy import deepcopy
from pathlib import PurePosixPath
import re
from typing import Any, Literal, TypeAlias

from pydantic import Field, field_validator, model_validator

from feedbax.contracts.figures import (
    FIGURE_DATA_PRODUCT_PAYLOAD_SCHEMA_ID,
    FIGURE_DATA_PRODUCT_PAYLOAD_SCHEMA_VERSION,
    FIGURE_INPUT_AUTHORITY_SCHEMA_ID,
    FIGURE_INPUT_AUTHORITY_SCHEMA_VERSION,
    FigureSpec,
)
from feedbax.contracts.manifest import (
    ParentRef,
    StrictModel,
    authenticated_manifest_ref_metadata,
)
from feedbax.contracts.row_index import (
    ResolvedRowSet,
    RowIndexCustodyBindings,
    RowSelectionError,
    RowSelectionErrorCode,
    RowSetSelector,
)

FIGURE_ROW_EXPANSION_REQUEST_SCHEMA_ID = "feedbax.spec.figure_row_expansion_request"
FIGURE_ROW_EXPANSION_REQUEST_SCHEMA_VERSION = "feedbax.spec.figure_row_expansion_request.v1"
RESOLVED_FIGURE_INPUTS_SCHEMA_ID = "feedbax.spec.resolved_figure_inputs"
RESOLVED_FIGURE_INPUTS_SCHEMA_VERSION = "feedbax.spec.resolved_figure_inputs.v1"
FIGURE_ROW_CUSTODY_LOCATOR_SCHEMA_ID = "feedbax.spec.figure_row_custody_locator"
FIGURE_ROW_CUSTODY_LOCATOR_SCHEMA_VERSION = "feedbax.spec.figure_row_custody_locator.v1"

ROW_NAMESPACE_TEMPLATE = "row_{ordinal}__"
PANEL_TITLE_SEPARATOR = "—"

#: Fields that only a run may decide. Authoring any of them is a hard failure.
AUTHORED_AUTHORITY_FIELDS = frozenset(
    {
        "authenticated_manifest",
        "manifest_sha256",
        "metadata",
        "ref_schema_id",
        "ref_schema_version",
        "sha256",
        "size_bytes",
    }
)


def validate_payload_schema_pair(
    payload_schema_id: str | None,
    payload_schema_version: str | None,
    *,
    input_role: str,
) -> None:
    """Refuse a half-stated decoded-payload schema identity.

    An artifact contract either pins the identity of the payload it decodes or
    states nothing about it. Half of the pair is neither: an id alone admits
    every version of that schema, and a version alone admits every schema that
    happens to spell its version the same way. Both models that carry the pair —
    the authored dialect contract and the runtime binding contract — refuse
    through here, so one rule states it once.

    Raises:
        ValueError: Exactly one of the two is stated, or one of them is blank.
    """
    stated = (payload_schema_id, payload_schema_version)
    if any(value is not None and not str(value).strip() for value in stated):
        raise ValueError(
            f"figure input role {input_role!r} states a blank payload schema identity; a "
            "stated payload schema is a nonempty schema id and a nonempty schema version"
        )
    if (payload_schema_id is None) != (payload_schema_version is None):
        raise ValueError(
            f"figure input role {input_role!r} states payload_schema_id and "
            "payload_schema_version together or states neither; half a payload schema "
            "identity pins nothing, because an id without a version admits every version "
            "of that schema"
        )


class FigureRoleReferenceError(ValueError):
    """One actionable rejection naming the offending role and field."""

    def __init__(self, message: str, *, input_role: str, field: str | None = None) -> None:
        super().__init__(message)
        self.input_role = input_role
        self.field = field


class PerRowInputReference(StrictModel):
    """A role resolved once per expanded row from the row index's custody."""

    per_row: str


class SharedInputReference(StrictModel):
    """An experiment-invariant role naming one manifest for every row."""

    shared: str


FigureInputReference: TypeAlias = PerRowInputReference | SharedInputReference


class FigureRoleBindingContract(StrictModel):
    """The declared, closed artifact contract for one figure input role.

    ``payload_schema_id`` and ``payload_schema_version`` pin the identity of the
    decoded JSON payload the selected artifact must carry. Figure execution
    already verifies both against the bytes it reads
    (:class:`~feedbax.contracts.figures.FigureArtifactPayload`), so stating them
    here is what lets a declaration reach that check instead of leaving it
    unstated. They are all-or-none: a schema id without a version pins half an
    identity, and half an identity admits every version of that schema.
    """

    input_role: str
    kind: str = "AnalysisRunManifest"
    authority: Literal["artifact_provider", "analysis_data_product"] = "artifact_provider"
    artifact_role: str
    artifact_provider: str
    manifest_status: Literal["completed"] = "completed"
    media_type: str = "application/json"
    payload_name: str | None = None
    payload_schema_id: str | None = None
    payload_schema_version: str | None = None
    product_role: str | None = None
    product_schema_id: str | None = None
    product_schema_version: str | None = None
    analysis_type: str | None = None

    @model_validator(mode="after")
    def _validate_contract(self) -> "FigureRoleBindingContract":
        if not self.input_role.strip():
            raise ValueError("FigureRoleBindingContract input_role must be nonempty")
        validate_payload_schema_pair(
            self.payload_schema_id,
            self.payload_schema_version,
            input_role=self.input_role,
        )
        product_fields = (self.product_role, self.product_schema_id, self.product_schema_version)
        if self.authority == "analysis_data_product":
            if any(value is None for value in product_fields):
                raise ValueError(
                    f"figure input role {self.input_role!r} declares an analysis data product "
                    "without a complete product_role/product_schema_id/product_schema_version"
                )
        elif any(value is not None for value in product_fields):
            raise ValueError(
                f"figure input role {self.input_role!r} declares product identity without "
                "authority 'analysis_data_product'"
            )
        return self

    @property
    def payload_key(self) -> str:
        """Return the row-invariant artifact payload name for this role."""
        return self.payload_name or self.input_role

    def artifact_payload(self, manifest_role: str) -> dict[str, Any]:
        """Return the fully derived artifact payload for one resolved input."""
        payload: dict[str, Any] = {
            "name": self.payload_key,
            "authority": self.authority,
            "manifest_role": manifest_role,
            "artifact_role": self.artifact_role,
            "artifact_provider": self.artifact_provider,
            "media_type": self.media_type,
            "manifest_status": self.manifest_status,
        }
        if self.payload_schema_id is not None:
            payload["payload_schema_id"] = self.payload_schema_id
            payload["payload_schema_version"] = self.payload_schema_version
        if self.authority == "analysis_data_product":
            payload.update(
                {
                    "schema_id": FIGURE_DATA_PRODUCT_PAYLOAD_SCHEMA_ID,
                    "schema_version": FIGURE_DATA_PRODUCT_PAYLOAD_SCHEMA_VERSION,
                    "product_role": self.product_role,
                    "product_schema_id": self.product_schema_id,
                    "product_schema_version": self.product_schema_version,
                }
            )
        return payload


def _validate_input_reference(role: str, payload: Any) -> FigureInputReference:
    """Coerce one authored role reference or fail with the offending field."""
    if isinstance(payload, (PerRowInputReference, SharedInputReference)):
        return payload
    if not isinstance(payload, Mapping):
        raise FigureRoleReferenceError(
            f"figure input role {role!r} must reference a role, not a literal value",
            input_role=role,
        )
    illegal = sorted(AUTHORED_AUTHORITY_FIELDS.intersection(payload))
    if illegal:
        raise FigureRoleReferenceError(
            f"figure input role {role!r} authors realization field {illegal[0]!r}; manifest "
            "authority is resolved at compile time and recorded in the compile lock, never "
            "authored",
            input_role=role,
            field=illegal[0],
        )
    keys = set(payload)
    if keys == {"per_row"}:
        return PerRowInputReference.model_validate(payload)
    if keys == {"shared"}:
        return SharedInputReference.model_validate(payload)
    raise FigureRoleReferenceError(
        f"figure input role {role!r} must declare exactly one of 'per_row' or 'shared', "
        f"not {sorted(keys)!r}",
        input_role=role,
    )


class FigureRowExpansionRequest(StrictModel):
    """One closed, versioned request to expand a figure over a row set.

    This document is compiler input, never hand-authored bytes: a downstream
    authoring layer builds it from its own envelope. It admits identity, the
    row-set selector, role references, and the declared role contracts. It
    admits no realization facts at all.
    """

    schema_id: str = FIGURE_ROW_EXPANSION_REQUEST_SCHEMA_ID
    schema_version: str = FIGURE_ROW_EXPANSION_REQUEST_SCHEMA_VERSION
    figure_name: str
    rows: RowSetSelector
    inputs: dict[str, FigureInputReference]
    role_contracts: list[FigureRoleBindingContract] = Field(min_length=1)
    assembler_title: str | None = None

    @field_validator("inputs", mode="before")
    @classmethod
    def _coerce_inputs(cls, value: Any) -> Any:
        if not isinstance(value, Mapping):
            return value
        return {role: _validate_input_reference(role, item) for role, item in value.items()}

    @model_validator(mode="after")
    def _validate_request(self) -> "FigureRowExpansionRequest":
        if self.schema_id != FIGURE_ROW_EXPANSION_REQUEST_SCHEMA_ID:
            raise ValueError(
                f"unsupported FigureRowExpansionRequest schema_id: {self.schema_id!r}"
            )
        if self.schema_version != FIGURE_ROW_EXPANSION_REQUEST_SCHEMA_VERSION:
            raise ValueError(
                f"unsupported FigureRowExpansionRequest schema_version: {self.schema_version!r}"
            )
        if not self.figure_name.strip():
            raise ValueError("FigureRowExpansionRequest figure_name must be nonempty")
        if not self.inputs:
            raise ValueError("FigureRowExpansionRequest declares no figure inputs")
        declared = [contract.input_role for contract in self.role_contracts]
        if len(declared) != len(set(declared)):
            raise ValueError("FigureRowExpansionRequest role contracts name a role twice")
        unknown = sorted(set(self.inputs) - set(declared))
        if unknown:
            raise FigureRoleReferenceError(
                f"figure input role {unknown[0]!r} is not declared by the base figure's "
                f"input binding contracts {sorted(declared)!r}",
                input_role=unknown[0],
            )
        return self

    def contract(self, input_role: str) -> FigureRoleBindingContract:
        """Return the declared contract for one input role or fail closed."""
        for contract in self.role_contracts:
            if contract.input_role == input_role:
                return contract
        raise FigureRoleReferenceError(
            f"figure input role {input_role!r} has no declared binding contract",
            input_role=input_role,
        )


class ResolvedFigureInput(StrictModel):
    """One expanded figure input and its compile-lock custody, if bound yet."""

    position: int = Field(ge=0)
    role: str
    input_role: str
    row_id: str
    row_ordinal: int = Field(ge=1)
    binding: Literal["per_row", "shared"]
    parent: ParentRef | None = None
    manifest_sha256: str | None = None
    size_bytes: int | None = None
    pending_reason: Literal["awaiting_run_receipt"] | None = None

    @model_validator(mode="after")
    def _validate_binding_state(self) -> "ResolvedFigureInput":
        bound = self.parent is not None
        if bound == (self.pending_reason is not None):
            raise ValueError(
                "ResolvedFigureInput must declare either a resolved parent or a pending reason"
            )
        if bound and self.parent is not None and self.parent.metadata:
            raise ValueError(
                "ResolvedFigureInput parent must stay free of authority metadata; digests "
                "belong in the compile lock"
            )
        if not bound and (self.manifest_sha256 is not None or self.size_bytes is not None):
            raise ValueError("pending ResolvedFigureInput cannot carry custody digests")
        return self


class ResolvedFigureInputs(StrictModel):
    """The full ordered expansion of figure roles over one resolved row set."""

    schema_id: str = RESOLVED_FIGURE_INPUTS_SCHEMA_ID
    schema_version: str = RESOLVED_FIGURE_INPUTS_SCHEMA_VERSION
    index_id: str
    index_sha256: str
    row_ids: list[str] = Field(min_length=1)
    inputs: list[ResolvedFigureInput] = Field(min_length=1)

    @model_validator(mode="after")
    def _validate_inputs(self) -> "ResolvedFigureInputs":
        if self.schema_id != RESOLVED_FIGURE_INPUTS_SCHEMA_ID:
            raise ValueError(f"unsupported ResolvedFigureInputs schema_id: {self.schema_id!r}")
        if self.schema_version != RESOLVED_FIGURE_INPUTS_SCHEMA_VERSION:
            raise ValueError(
                f"unsupported ResolvedFigureInputs schema_version: {self.schema_version!r}"
            )
        if [item.position for item in self.inputs] != list(range(len(self.inputs))):
            raise ValueError("ResolvedFigureInputs positions must be contiguous and ordered")
        roles = [item.role for item in self.inputs]
        if len(roles) != len(set(roles)):
            raise ValueError("ResolvedFigureInputs role names must be unique")
        return self

    @property
    def pending_roles(self) -> tuple[str, ...]:
        """Return the roles still awaiting a post-run custody binding."""
        return tuple(item.role for item in self.inputs if item.pending_reason is not None)

    @property
    def fully_bound(self) -> bool:
        """Whether every expanded role has authenticated custody."""
        return not self.pending_roles


class FigureRowCustodyLocator(StrictModel):
    """Where one row-expanded figure's per-row custody bindings are to be found.

    A ``per_row`` role is filled from the row index's post-run custody, and that
    custody lives in a :class:`~feedbax.contracts.row_index.RowIndexCustodyBindings`
    document the receipt layer produces. The compile cannot pin that document's
    bytes — they do not exist until the rows have run — and it must not derive
    its location by convention from the index id, because a convention is a rule
    nothing states. So the declaration *names* the document, the compile records
    that name here, and fulfillment resolves it once the rows have been produced.

    This record carries a locator and the identity the located document must
    match. It carries no digest of the custody document itself: that is a
    realization fact, and pinning it would invalidate the compile every time
    custody moved.

    Attributes:
        index_id: The row index the located custody must belong to.
        index_sha256: The digest of the row index the rows were expanded against,
            so custody for a re-cut index cannot bind to this expansion.
        ref: The repo-relative path of the custody bindings document.
        binding_keys: The per-row binding keys this figure's roles require, in
            sorted order. A located document missing one of them for any expanded
            row fails closed rather than binding part of the figure.
    """

    schema_id: str = FIGURE_ROW_CUSTODY_LOCATOR_SCHEMA_ID
    schema_version: str = FIGURE_ROW_CUSTODY_LOCATOR_SCHEMA_VERSION
    index_id: str
    index_sha256: str
    ref: str
    binding_keys: list[str] = Field(min_length=1)

    @model_validator(mode="after")
    def _validate_locator(self) -> "FigureRowCustodyLocator":
        if self.schema_id != FIGURE_ROW_CUSTODY_LOCATOR_SCHEMA_ID:
            raise ValueError(
                f"unsupported FigureRowCustodyLocator schema_id: {self.schema_id!r}"
            )
        if self.schema_version != FIGURE_ROW_CUSTODY_LOCATOR_SCHEMA_VERSION:
            raise ValueError(
                f"unsupported FigureRowCustodyLocator schema_version: {self.schema_version!r}"
            )
        if not self.index_id.strip():
            raise ValueError("FigureRowCustodyLocator index_id must be nonempty")
        if not re.fullmatch(r"[0-9a-f]{64}", self.index_sha256):
            raise ValueError("FigureRowCustodyLocator index_sha256 must be lowercase sha256")
        _validate_custody_ref(self.ref)
        if len(self.binding_keys) != len(set(self.binding_keys)):
            raise ValueError("FigureRowCustodyLocator names a binding key twice")
        if list(self.binding_keys) != sorted(self.binding_keys):
            raise ValueError(
                "FigureRowCustodyLocator binding keys are sorted, so one declaration always "
                "produces one record"
            )
        return self


def _validate_custody_ref(ref: str) -> str:
    """Refuse a custody ref that addresses anything but a repo-relative file."""
    if not ref.strip():
        raise ValueError("FigureRowCustodyLocator ref must be nonempty")
    candidate = PurePosixPath(ref)
    if candidate.is_absolute() or ref.startswith("\\"):
        raise ValueError(
            f"figure row custody ref {ref!r} is absolute; a custody document is addressed "
            "repo-relatively so one repository resolves it the same way everywhere"
        )
    if ".." in candidate.parts:
        raise ValueError(
            f"figure row custody ref {ref!r} escapes the repository root; a custody document "
            "is a tracked repository path"
        )
    return ref


def per_row_binding_keys(request: "FigureRowExpansionRequest") -> tuple[str, ...]:
    """Return the sorted per-row binding keys one expansion request requires."""
    return tuple(
        sorted(
            {
                reference.per_row
                for reference in request.inputs.values()
                if isinstance(reference, PerRowInputReference)
            }
        )
    )


def row_namespace(ordinal: int) -> str:
    """Return the ``row_{n}__`` namespace prefix for a 1-based row ordinal."""
    return ROW_NAMESPACE_TEMPLATE.format(ordinal=ordinal)


def resolve_figure_input_roles(
    request: FigureRowExpansionRequest,
    resolved_rows: ResolvedRowSet,
    bindings: RowIndexCustodyBindings | None = None,
) -> ResolvedFigureInputs:
    """Expand role references over *resolved_rows* into ordered figure inputs.

    Expansion order is row-major: every declared role, in declaration order,
    for row 1, then for row 2, and so on. Role names are ``row_{n}__{role}``
    with ``n`` 1-based over the resolved row order.

    ``bindings`` is the post-run custody half. When it is omitted, per-row
    roles resolve to a pending entry rather than failing, so a first-time
    figure compiles before its rows have ever been produced. Shared roles are
    named directly by the author and bind either way.
    """
    if bindings is not None and bindings.index_id != resolved_rows.index_id:
        raise RowSelectionError(
            RowSelectionErrorCode.INDEX_MISMATCH,
            f"row custody bindings for {bindings.index_id!r} do not belong to resolved row "
            f"set {resolved_rows.index_id!r}",
            index_id=resolved_rows.index_id,
        )
    inputs: list[ResolvedFigureInput] = []
    for ordinal, row_id in enumerate(resolved_rows.row_ids, start=1):
        namespace = row_namespace(ordinal)
        for input_role, reference in request.inputs.items():
            contract = request.contract(input_role)
            role = f"{namespace}{input_role}"
            if isinstance(reference, SharedInputReference):
                inputs.append(
                    ResolvedFigureInput(
                        position=len(inputs),
                        role=role,
                        input_role=input_role,
                        row_id=row_id,
                        row_ordinal=ordinal,
                        binding="shared",
                        parent=ParentRef(kind=contract.kind, id=reference.shared, role=role),
                    )
                )
                continue
            if bindings is None:
                inputs.append(
                    ResolvedFigureInput(
                        position=len(inputs),
                        role=role,
                        input_role=input_role,
                        row_id=row_id,
                        row_ordinal=ordinal,
                        binding="per_row",
                        pending_reason="awaiting_run_receipt",
                    )
                )
                continue
            binding = bindings.resolve(row_id, reference.per_row)
            digest, size_bytes = binding.authenticated_profile()
            if binding.parent.kind != contract.kind:
                raise FigureRoleReferenceError(
                    f"row {row_id!r} binding {reference.per_row!r} is a "
                    f"{binding.parent.kind!r}, but role {input_role!r} declares "
                    f"{contract.kind!r}",
                    input_role=input_role,
                )
            inputs.append(
                ResolvedFigureInput(
                    position=len(inputs),
                    role=role,
                    input_role=input_role,
                    row_id=row_id,
                    row_ordinal=ordinal,
                    binding="per_row",
                    parent=ParentRef(kind=binding.parent.kind, id=binding.parent.id, role=role),
                    manifest_sha256=digest,
                    size_bytes=size_bytes,
                )
            )
    return ResolvedFigureInputs(
        index_id=resolved_rows.index_id,
        index_sha256=resolved_rows.index_sha256,
        row_ids=list(resolved_rows.row_ids),
        inputs=inputs,
    )


def _prefix_items(value: Any, namespace: str) -> Any:
    """Prefix every ``item`` reference beneath a trace's data declaration."""
    if isinstance(value, Mapping):
        prefixed = {key: _prefix_items(item, namespace) for key, item in value.items()}
        if isinstance(value.get("item"), str):
            prefixed["item"] = f"{namespace}{value['item']}"
        return prefixed
    if isinstance(value, list):
        return [_prefix_items(item, namespace) for item in value]
    return value


def _expanded_family(family: Mapping[str, Any], namespace: str, *, owns_legend: bool) -> dict:
    expanded = deepcopy(dict(family))
    expanded["name"] = f"{namespace}{family['name']}"
    trace = deepcopy(dict(family["trace"]))
    trace["name"] = f"{namespace}{trace['name']}"
    if isinstance(trace.get("panel"), str):
        trace["panel"] = f"{namespace}{trace['panel']}"
    if "data" in trace:
        trace["data"] = _prefix_items(trace["data"], namespace)
    if not owns_legend and "legend_index" in expanded:
        expanded.pop("legend_index")
        params = dict(trace.get("params") or {})
        params["showlegend"] = False
        trace["params"] = params
    expanded["trace"] = trace
    return expanded


def _expanded_panel(
    panel: Mapping[str, Any],
    namespace: str,
    *,
    row_label: str,
    row_offset: int,
) -> dict:
    expanded = deepcopy(dict(panel))
    expanded["name"] = f"{namespace}{panel['name']}"
    if isinstance(panel.get("title"), str):
        expanded["title"] = f"{row_label} {PANEL_TITLE_SEPARATOR} {panel['title']}"
    if isinstance(panel.get("row"), int):
        expanded["row"] = panel["row"] + row_offset
    return expanded


def _expand_figure_rows_structure(
    base_figure: FigureSpec | Mapping[str, Any],
    request: FigureRowExpansionRequest,
    resolved_rows: ResolvedRowSet,
) -> dict[str, Any]:
    """Derive the multi-row figure payload, leaving its inputs unbound.

    This is the half of the expansion that a *plan* can state: namespaces, panel
    placement and titles, legend ownership, colorbar placement, assembler height
    and title. Nothing here names produced data, so it is derivable before any
    row has ever run. :func:`expand_figure_rows` is this plus the post-run
    custody binding.

    Internal. The supported surfaces are :func:`expand_figure_rows`, which
    derives the whole bound figure, and the built-in envelope compiler, which is
    what produces a row-expanded figure's compiled document. The structural half
    alone is an implementation detail of both: it is neither versioned nor part
    of the downstream stability contract, and it may change with the derivation
    rules it implements.
    """
    base = (
        base_figure.model_dump(mode="json", exclude_none=True)
        if isinstance(base_figure, FigureSpec)
        else deepcopy(dict(base_figure))
    )
    payload = deepcopy(base)
    payload["name"] = request.figure_name
    row_count = len(resolved_rows.row_ids)
    first_namespace = row_namespace(1)

    base_panels: Sequence[Mapping[str, Any]] = base.get("panels") or ()
    if not base_panels:
        raise ValueError("figure row expansion requires a base figure that declares panels")
    row_block = max(int(panel.get("row", 1)) for panel in base_panels)
    base_families: Sequence[Mapping[str, Any]] = base.get("trace_families") or ()
    if not base_families:
        raise ValueError(
            "figure row expansion requires a base figure that declares trace families; the "
            "base is the single-row scientific statement being repeated per row"
        )
    unsupported = sorted(
        key for key in ("traces", "slot_bindings", "slot_families") if base.get(key)
    )
    if unsupported:
        raise ValueError(
            f"figure row expansion does not know how to repeat base {unsupported} per row; "
            "expressing the base through trace families keeps one derivation rule"
        )

    panels: list[dict[str, Any]] = []
    families: list[dict[str, Any]] = []
    for ordinal, label in enumerate(
        resolved_rows.labels, start=1
    ):
        namespace = row_namespace(ordinal)
        for panel in base_panels:
            panels.append(
                _expanded_panel(
                    panel,
                    namespace,
                    row_label=label,
                    row_offset=(ordinal - 1) * row_block,
                )
            )
        for family in base_families:
            families.append(_expanded_family(family, namespace, owns_legend=ordinal == 1))
    payload["panels"] = panels
    if base_families:
        payload["trace_families"] = families

    assembler_params = dict(base.get("assembler_params") or {})
    if "height" in assembler_params:
        assembler_params["height"] = assembler_params["height"] * row_count
    if request.assembler_title is not None:
        assembler_params["title"] = request.assembler_title
    payload["assembler_params"] = assembler_params

    colorbar = base.get("colorbar")
    if isinstance(colorbar, Mapping):
        colorbar = deepcopy(dict(colorbar))
        if isinstance(colorbar.get("family"), str):
            colorbar["family"] = f"{first_namespace}{colorbar['family']}"
        placement = colorbar.get("placement")
        if isinstance(placement, Mapping) and isinstance(placement.get("panel"), str):
            placement = dict(placement)
            placement["panel"] = f"{first_namespace}{placement['panel']}"
            colorbar["placement"] = placement
        payload["colorbar"] = colorbar
    return payload


def expand_figure_rows(
    base_figure: FigureSpec | Mapping[str, Any],
    request: FigureRowExpansionRequest,
    resolved_rows: ResolvedRowSet,
    resolved_inputs: ResolvedFigureInputs,
) -> FigureSpec:
    """Derive the multi-row :class:`FigureSpec` for one resolved row set.

    The base figure is the single-row scientific statement; nothing about the
    expansion is authored. Row-index order alone determines the ``row_{n}__``
    namespace, panel placement and titles, legend ownership, colorbar
    placement, and assembler height.

    Raises:
        RowSelectionError: If any expanded role still awaits a run receipt, so
            the figure would name data that has not been produced.
    """
    if not resolved_inputs.fully_bound:
        raise RowSelectionError(
            RowSelectionErrorCode.UNRESOLVED_ROW_KEY,
            "figure expansion requires post-run custody for every per-row role; still "
            f"pending: {list(resolved_inputs.pending_roles)}",
            index_id=resolved_inputs.index_id,
        )
    if list(resolved_inputs.row_ids) != list(resolved_rows.row_ids):
        raise RowSelectionError(
            RowSelectionErrorCode.INDEX_MISMATCH,
            "resolved figure inputs disagree with the resolved row set",
            index_id=resolved_rows.index_id,
        )
    payload = _expand_figure_rows_structure(base_figure, request, resolved_rows)
    inputs, authorities = figure_input_binding_records(request, resolved_inputs.inputs)
    payload["inputs"] = list(inputs)
    payload["input_authorities"] = list(authorities)
    return FigureSpec.model_validate(payload)


def figure_input_binding_records(
    request: FigureRowExpansionRequest,
    inputs: Sequence[ResolvedFigureInput],
    *,
    authenticated: bool = False,
) -> tuple[tuple[dict[str, Any], ...], tuple[dict[str, Any], ...]]:
    """Return the ``(inputs, input_authorities)`` records *inputs* bind to.

    This is the one derivation of a figure's bound inputs from resolved roles.
    :func:`expand_figure_rows` uses it over a whole expansion to build a figure's
    own document; fulfillment uses it over the per-row subset to build the
    runtime overlay that binds an already-expanded compiled figure, which is why
    it takes a sequence of resolved inputs rather than the whole resolution.

    Each authority quotes the same parent record as the input it authorizes, so
    the two always address one manifest, and the artifact payload contract comes
    from the role's declared contract rather than from anything the caller says.

    ``authenticated`` selects which of the two records is being built, and the
    two differ in exactly one way. A figure *document* carries no digests: they
    are realization facts, they belong in the compile lock, and putting them in
    the document would invalidate the figure's identity every time custody
    moved. A *runtime overlay* is the opposite case — it lives outside figure
    identity and is what execution authenticates against, so it restates the
    custody profile the resolution already carries
    (:attr:`ResolvedFigureInput.manifest_sha256` and
    :attr:`ResolvedFigureInput.size_bytes`) as the parent's authenticated
    manifest ref metadata. Nothing is invented: the profile comes from the
    custody document that produced the bytes, and an input that carries no
    profile is refused rather than bound unauthenticated.

    Raises:
        RowSelectionError: If any resolved input is still awaiting a run receipt,
            so binding it would name data that has not been produced, or if
            ``authenticated`` is set and an input carries no custody profile.
    """
    pending = [item.role for item in inputs if item.parent is None]
    if pending:
        raise RowSelectionError(
            RowSelectionErrorCode.UNRESOLVED_ROW_KEY,
            "figure input binding requires post-run custody for every bound role; still "
            f"pending: {pending}",
        )
    bound: list[dict[str, Any]] = []
    authorities: list[dict[str, Any]] = []
    for item in inputs:
        assert item.parent is not None  # guaranteed by the pending check
        parent = item.parent.model_dump(mode="json", exclude_none=True)
        parent.pop("metadata", None)
        if authenticated:
            if item.manifest_sha256 is None or item.size_bytes is None:
                raise RowSelectionError(
                    RowSelectionErrorCode.UNRESOLVED_ROW_KEY,
                    f"figure input role {item.role!r} binds {item.parent.id!r} with no "
                    "authenticated custody profile; a runtime overlay states the digest and "
                    "byte size the custody document recorded, and an unauthenticated parent "
                    "is refused rather than bound",
                    row_id=item.row_id,
                )
            parent["metadata"] = authenticated_manifest_ref_metadata(
                item.manifest_sha256, item.size_bytes
            )
        bound.append(parent)
        contract = request.contract(item.input_role)
        authorities.append(
            {
                "schema_id": FIGURE_INPUT_AUTHORITY_SCHEMA_ID,
                "schema_version": FIGURE_INPUT_AUTHORITY_SCHEMA_VERSION,
                "parent": parent,
                "artifact_payloads": [contract.artifact_payload(item.role)],
            }
        )
    return tuple(bound), tuple(authorities)


__all__ = [
    "AUTHORED_AUTHORITY_FIELDS",
    "FIGURE_ROW_CUSTODY_LOCATOR_SCHEMA_ID",
    "FIGURE_ROW_CUSTODY_LOCATOR_SCHEMA_VERSION",
    "FIGURE_ROW_EXPANSION_REQUEST_SCHEMA_ID",
    "FIGURE_ROW_EXPANSION_REQUEST_SCHEMA_VERSION",
    "RESOLVED_FIGURE_INPUTS_SCHEMA_ID",
    "RESOLVED_FIGURE_INPUTS_SCHEMA_VERSION",
    "FigureInputReference",
    "FigureRoleBindingContract",
    "FigureRoleReferenceError",
    "FigureRowCustodyLocator",
    "FigureRowExpansionRequest",
    "PerRowInputReference",
    "ResolvedFigureInput",
    "ResolvedFigureInputs",
    "SharedInputReference",
    "expand_figure_rows",
    "figure_input_binding_records",
    "per_row_binding_keys",
    "resolve_figure_input_roles",
    "row_namespace",
    "validate_payload_schema_pair",
]
