"""Binding a row-expanded figure's per-row roles from produced custody.

A row-expanded figure is compiled as *structure*: the ``row_{n}__`` namespaces,
the panel placement and titles, the legend ownership, the assembler height. What
it deliberately does not carry is which produced artifact fills each expanded
row's roles, because authenticating one takes a run and a compile is a plan.

The compile therefore records three facts and stops: the expansion request
(:class:`~feedbax.contracts.figure_roles.FigureRowExpansionRequest`), the rows it
resolved (:class:`~feedbax.contracts.row_index.ResolvedRowSet`), and — when any
role is filled per row — where the custody bindings for those rows will be
(:class:`~feedbax.contracts.figure_roles.FigureRowCustodyLocator`). This module
is the other end: at fulfillment, once the rows have actually been produced, it
locates that custody document, proves it belongs to this expansion, and turns it
into the runtime overlay the figure executes with.

## Where custody lands, and where it does not

The bound parents and their input authorities are a *runtime overlay*
(:attr:`~feedbax.analysis.fulfillment_adapters.FigureNodeRequest.runtime_inputs`),
never an edit to the compiled document. Digests and byte sizes stay outside the
figure's scientific identity, which is the same boundary the compile drew — and
inside the overlay, which is what execution authenticates against. Each bound
parent therefore restates the custody document's own ``(sha256, size_bytes)``
profile as authenticated manifest ref metadata, so figure execution resolves the
exact bytes custody recorded instead of refusing a parent it cannot authenticate.

## The cut is proved, not asserted

A custody document names the row index it belongs to, and an index id is a name
two cuts of one index both wear. So before any binding, the row index the
expansion resolved against is re-read from the declared repository and its
canonical digest re-derived; it must equal the digest the lock pinned, and the
located custody document's bindings must belong to that proved cut. A stale
document from another cut is refused rather than bound row by row.

## Both refs are resolved inside the declared repository

The row index and the custody document are both named repo-relatively, and a
repo-relative name is a claim about *which* repository holds the bytes. Joining
one onto ``repo_root`` does not by itself keep it there: an absolute ref replaces
the root outright, an upward ref walks past it, and a symlinked directory
component leaves it without either. Every one of those reaches a document the
declared repository does not hold, and the digest checks downstream would not
notice, because they compare bytes with bytes and never ask whose repository the
bytes came from. A right document authenticated under the wrong repository
identity is the failure containment exists to refuse, so both joins resolve
symlinks and require the result to stay under the resolved root.

## Only per-row roles come from here

A ``shared`` role names one manifest for every row, states an ordinary reference
in the envelope, and is therefore bound by the ordinary plan-edge path from an
admitted receipt. Binding it a second time from the authored key would be two
bindings that can disagree, so this module contributes the per-row roles and
nothing else.

## Every failure is a refusal

An absent locator, an absent document, a document belonging to another index or
another cut of the same index, a row or binding key the document does not carry,
and a role left pending after resolution are each refused by name. None of them
is an inapplicability: a per-row role the declaration states is required, and a
figure that rendered without it would silently be a different figure.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
import json
from pathlib import Path, PurePosixPath
from typing import Any

from feedbax.analysis.fulfillment_derivation import (
    CompiledEnvelope,
    FulfillmentDerivationError,
)
from feedbax.contracts.figures import FigureInputAuthority, FigureInputAuthoritySpec
from feedbax.contracts.figure_roles import (
    FigureRoleReferenceError,
    FigureRowCustodyLocator,
    FigureRowExpansionRequest,
    figure_input_binding_records,
    per_row_binding_keys,
    resolve_figure_input_roles,
)
from feedbax.contracts.manifest import ParentRef
from feedbax.contracts.row_index import (
    AuthenticatedRowIndex,
    ResolvedRowSet,
    RowIndexCustodyBindings,
    RowSelectionError,
    load_row_index_custody_bindings,
)

#: The identity-contribution keys a row-expanded figure's compile records. They
#: are the compiler's own naming, restated here so a reader can find them without
#: a compiler.
ROW_EXPANSION_CONTRIBUTION = "figure_row_expansion"
RESOLVED_ROW_SET_CONTRIBUTION = "resolved_row_set"
ROW_CUSTODY_CONTRIBUTION = "row_custody"


class RowCustodyFulfillmentError(FulfillmentDerivationError):
    """A row-expanded figure's per-row custody cannot be located or believed.

    Every instance names the compiled node it refuses, so the refusal points at a
    declaration or a missing production rather than at this module.
    """

    def __init__(self, message: str, *, ref: str, custody_ref: str | None = None) -> None:
        self.ref = ref
        self.custody_ref = custody_ref
        super().__init__(f"{ref}: {message}")


@dataclass(frozen=True)
class RowExpansionRecord:
    """What one compiled figure's lock says about its row expansion.

    ``locator`` is ``None`` exactly when the expansion fills no role per row, in
    which case there is no custody to find.
    """

    request: FigureRowExpansionRequest
    resolved_rows: ResolvedRowSet
    locator: FigureRowCustodyLocator | None

    @property
    def per_row_keys(self) -> tuple[str, ...]:
        """Return the per-row binding keys this expansion's roles require."""
        return per_row_binding_keys(self.request)


@dataclass(frozen=True)
class RowCustodyOverlay:
    """The runtime overlay one figure's per-row custody binds.

    Attributes:
        inputs: The authenticated per-row parents, in expansion order.
        authorities: One input authority per bound parent, in the same order.
        bindings: The custody document the overlay was derived from.
        path: Where that document was read from.
    """

    inputs: tuple[ParentRef, ...]
    authorities: tuple[FigureInputAuthoritySpec, ...]
    bindings: RowIndexCustodyBindings
    path: Path


def read_row_expansion_record(compiled: CompiledEnvelope) -> RowExpansionRecord | None:
    """Return what one compiled output's lock states about its row expansion.

    Returns ``None`` for every compiled output that is not a row-expanded figure,
    which is how a caller distinguishes "this node has no per-row custody" from
    "this node's per-row custody is broken" without inspecting layers itself.

    Raises:
        RowCustodyFulfillmentError: The lock carries a row-expansion contribution
            that is incomplete or does not validate. A half-recorded expansion is
            a broken lock, not an absent one.
    """
    ref = str(compiled.lock_path)
    contributions = compiled.lock.get("identity_contributions")
    if not isinstance(contributions, Mapping) or ROW_EXPANSION_CONTRIBUTION not in contributions:
        return None
    request = _validated(
        FigureRowExpansionRequest,
        contributions.get(ROW_EXPANSION_CONTRIBUTION),
        ref=ref,
        what=ROW_EXPANSION_CONTRIBUTION,
    )
    if RESOLVED_ROW_SET_CONTRIBUTION not in contributions:
        raise RowCustodyFulfillmentError(
            f"records a {ROW_EXPANSION_CONTRIBUTION!r} identity contribution but no "
            f"{RESOLVED_ROW_SET_CONTRIBUTION!r}; a row expansion is the request and the rows "
            "it was expanded against, and one without the other addresses nothing",
            ref=ref,
        )
    resolved_rows = _validated(
        ResolvedRowSet,
        contributions.get(RESOLVED_ROW_SET_CONTRIBUTION),
        ref=ref,
        what=RESOLVED_ROW_SET_CONTRIBUTION,
    )
    locator = (
        _validated(
            FigureRowCustodyLocator,
            contributions.get(ROW_CUSTODY_CONTRIBUTION),
            ref=ref,
            what=ROW_CUSTODY_CONTRIBUTION,
        )
        if ROW_CUSTODY_CONTRIBUTION in contributions
        else None
    )
    return RowExpansionRecord(request=request, resolved_rows=resolved_rows, locator=locator)


def resolve_row_custody_overlay(
    compiled: CompiledEnvelope, *, repo_root: Path | str | None
) -> RowCustodyOverlay | None:
    """Bind one compiled row-expanded figure's per-row roles, or refuse.

    Returns ``None`` when the compiled output is not a row-expanded figure, or is
    one whose every role is shared, because neither has a per-row role to fill.
    Every other outcome is either a complete overlay or a named refusal: there is
    no partial binding and no fallback to the pending state the compile recorded,
    which would render a figure missing the data it exists to show.

    Args:
        compiled: The lock/document pair the figure node executes.
        repo_root: The trusted repository root the custody ref is resolved
            against. A row-expanded figure whose environment declares none is
            refused rather than resolved against the process working directory.

    Raises:
        RowCustodyFulfillmentError: The custody document is undeclared, absent,
            unreadable, foreign to this expansion, incomplete for it, or reached
            by a ref that leaves the declared repository.
    """
    record = read_row_expansion_record(compiled)
    if record is None:
        return None
    ref = str(compiled.lock_path)
    keys = record.per_row_keys
    if not keys:
        if record.locator is not None:
            raise RowCustodyFulfillmentError(
                "records a per-row custody locator while every input role is shared; a "
                "shared role names its own manifest and binds no row custody",
                ref=ref,
            )
        return None
    locator = record.locator
    if locator is None:
        raise RowCustodyFulfillmentError(
            f"fills the role(s) {list(keys)} once per expanded row from the row index's "
            "custody, but its compile lock names no custody bindings document. State the "
            "document in the figure envelope as 'row_custody' and recompile: a per-row role "
            "that nothing addresses cannot be bound, and rendering the figure without it "
            "would silently be a different figure",
            ref=ref,
        )
    _require_locator_matches(locator, record.resolved_rows, ref=ref)
    index = _authenticated_row_index(record, locator, repo_root=repo_root, ref=ref)
    bindings, path = _load_custody(locator, repo_root=repo_root, ref=ref)
    _require_custody_belongs_to_cut(bindings, index, locator=locator, ref=ref)
    resolved = _resolve_inputs(record, bindings, ref=ref, custody_ref=locator.ref)
    per_row = tuple(item for item in resolved.inputs if item.binding == "per_row")
    inputs, authorities = figure_input_binding_records(
        record.request, per_row, authenticated=True
    )
    return RowCustodyOverlay(
        inputs=tuple(ParentRef.model_validate(item) for item in inputs),
        authorities=tuple(FigureInputAuthority.model_validate(item) for item in authorities),
        bindings=bindings,
        path=path,
    )


def _validated(model: type, payload: Any, *, ref: str, what: str):
    try:
        return model.model_validate(payload)
    except (ValueError, TypeError) as exc:
        raise RowCustodyFulfillmentError(
            f"records an identity contribution {what!r} that is not a "
            f"{model.__name__}: {exc}",
            ref=ref,
        ) from exc


def _repo_contained_path(ref: str, repo_root: Path | str) -> Path:
    """Return the path *ref* names beneath *repo_root*, or refuse to name one.

    A repo-relative ref addresses one repository's bytes, so containment is part
    of what the ref means rather than a precaution taken around it. Three ways
    out are refused before any read: an absolute ref, which discards the root on
    join; an upward ref, which walks past it; and a resolved location outside the
    root, which is what a symlinked directory component produces while every
    lexical part of the ref still looks repo-relative.

    The returned path is the *resolved* one, so the bytes that are read are the
    bytes containment was proved for. Resolution is non-strict: an absent
    document resolves and is then refused by the read that finds nothing, which
    is a different fact from a document the repository does not hold.

    Raises:
        ValueError: The ref is empty, absolute, upward, or resolves outside
            *repo_root*. The message states which, as a sentence fragment the
            caller completes with what the ref was being read for.
    """
    if not ref.strip():
        raise ValueError("it names an empty path")
    if PurePosixPath(ref).is_absolute() or Path(ref).is_absolute() or ref.startswith("\\"):
        raise ValueError("it is an absolute path rather than a repo-relative one")
    if ".." in PurePosixPath(ref).parts or ".." in Path(ref).parts:
        raise ValueError("it traverses upward out of the repository")
    root = Path(repo_root).resolve(strict=False)
    resolved = (root / ref).resolve(strict=False)
    if not resolved.is_relative_to(root):
        raise ValueError(
            f"it resolves to {resolved}, which is outside the repository root {root}; a "
            "symlinked path component leaves the repository while every part of the ref "
            "still reads as repo-relative"
        )
    return resolved


def _require_locator_matches(
    locator: FigureRowCustodyLocator, resolved_rows: ResolvedRowSet, *, ref: str
) -> None:
    """Prove the recorded locator addresses the rows this figure expanded over."""
    if locator.index_id != resolved_rows.index_id:
        raise RowCustodyFulfillmentError(
            f"names custody for row index {locator.index_id!r} while it expanded over "
            f"{resolved_rows.index_id!r}; one lock cannot describe two row sets",
            ref=ref,
            custody_ref=locator.ref,
        )
    if locator.index_sha256 != resolved_rows.index_sha256:
        raise RowCustodyFulfillmentError(
            f"names custody pinned to row index digest {locator.index_sha256} while its "
            f"resolved rows were expanded against {resolved_rows.index_sha256}; the lock "
            "describes two cuts of one index and neither can be trusted for the other",
            ref=ref,
            custody_ref=locator.ref,
        )


def _authenticated_row_index(
    record: RowExpansionRecord,
    locator: FigureRowCustodyLocator,
    *,
    repo_root: Path | str | None,
    ref: str,
) -> AuthenticatedRowIndex:
    """Return the exact index cut the lock pinned, re-derived from located bytes.

    ``index_sha256`` is the only lock-side digest a row expansion has, and until
    something recomputes it from bytes on disk it is compared with nothing but
    another copy of itself. This reads the row index the expansion names and
    re-derives its canonical digest, so the cut a custody document is checked
    against is a proved cut rather than an asserted one.

    Raises:
        RowCustodyFulfillmentError: The lock names no index to re-read, the
            declared repository is absent, the index is missing or does not
            validate, or its bytes are a different cut from the pinned one.
    """
    index_ref = record.resolved_rows.index_ref
    if not index_ref:
        raise RowCustodyFulfillmentError(
            "binds per-row custody against row index digest "
            f"{locator.index_sha256} but its resolved row set names no index_ref, so the "
            "cut the custody must belong to cannot be re-read and proved. Recompile: an "
            "index digest nothing can be checked against authenticates nothing",
            ref=ref,
            custody_ref=locator.ref,
        )
    if repo_root is None:
        raise RowCustodyFulfillmentError(
            f"authenticates its row index cut from {index_ref!r}, which is a repo-relative "
            "path, but the fulfillment environment declares no repo_root",
            ref=ref,
            custody_ref=locator.ref,
        )
    try:
        path = _repo_contained_path(index_ref, repo_root)
    except ValueError as exc:
        raise RowCustodyFulfillmentError(
            f"authenticates its row index cut from {index_ref!r}, which is not a path inside "
            f"the declared repository: {exc}. The cut is proved from the index's own bytes, "
            "and a valid index reached outside the declared repository would prove a cut of "
            "some other repository's index under this one's identity",
            ref=ref,
            custody_ref=locator.ref,
        ) from exc
    try:
        index = AuthenticatedRowIndex.model_validate(
            json.loads(path.read_text(encoding="utf-8"))
        )
    except (OSError, ValueError, TypeError) as exc:
        raise RowCustodyFulfillmentError(
            f"expanded over the row index {index_ref!r}, which does not read at {path} as an "
            f"AuthenticatedRowIndex: {exc}. The cut custody must belong to is proved from "
            "the index's own bytes, so an unreadable index is a refusal",
            ref=ref,
            custody_ref=locator.ref,
        ) from exc
    digest = index.canonical_sha256()
    if digest != locator.index_sha256:
        raise RowCustodyFulfillmentError(
            f"expanded over the row index {index_ref!r} pinned at {locator.index_sha256}, but "
            f"the index at {path} now canonicalizes to {digest}; the rows were resolved "
            "against a different cut, and custody bound to this one addresses other rows",
            ref=ref,
            custody_ref=locator.ref,
        )
    return index


def _require_custody_belongs_to_cut(
    bindings: RowIndexCustodyBindings,
    index: AuthenticatedRowIndex,
    *,
    locator: FigureRowCustodyLocator,
    ref: str,
) -> None:
    """Prove the located custody document's content belongs to the pinned cut.

    Sharing an ``index_id`` proves nothing: an id is a name, and two cuts of one
    index wear the same one. This checks the document's actual bindings against
    the authenticated index, so custody produced for another cut is refused
    rather than bound row by row.
    """
    try:
        bindings.require_index(index)
    except RowSelectionError as exc:
        raise RowCustodyFulfillmentError(
            f"binds per-row custody from {locator.ref!r}, whose bindings do not belong to the "
            f"row index cut pinned at {locator.index_sha256}: {exc}. A custody document is "
            "believed for the cut it was produced against, never for one that reuses its "
            "index id",
            ref=ref,
            custody_ref=locator.ref,
        ) from exc


def _load_custody(
    locator: FigureRowCustodyLocator, *, repo_root: Path | str | None, ref: str
) -> tuple[RowIndexCustodyBindings, Path]:
    """Read the custody document one locator names, or refuse by name."""
    if repo_root is None:
        raise RowCustodyFulfillmentError(
            f"binds per-row custody from {locator.ref!r}, which is a repo-relative path, but "
            "the fulfillment environment declares no repo_root. A custody document is "
            "resolved against a declared repository, never against a working directory",
            ref=ref,
            custody_ref=locator.ref,
        )
    try:
        path = _repo_contained_path(locator.ref, repo_root)
    except ValueError as exc:
        raise RowCustodyFulfillmentError(
            f"binds per-row custody from {locator.ref!r}, which is not a path inside the "
            f"declared repository: {exc}. A custody document is believed for the repository "
            "that holds it, and one reached outside the declared root could be a perfectly "
            "valid document belonging to another repository entirely — which the digest "
            "checks would not notice, because they compare bytes and never ask whose "
            "repository the bytes came from",
            ref=ref,
            custody_ref=locator.ref,
        ) from exc
    # One load, and the two outcomes it can have. Probing for the file and then
    # opening it is two looks at one path for no gain: absence is what the open
    # reports, and the distinction the two refusals draw is between "nothing is
    # there" and "something is there that is not this document", which the one
    # read decides.
    try:
        bindings = load_row_index_custody_bindings(path)
    except (FileNotFoundError, IsADirectoryError, NotADirectoryError) as exc:
        raise RowCustodyFulfillmentError(
            f"binds per-row custody from {locator.ref!r}, which is not a file at {path}. The "
            "custody bindings document is produced by the run receipt layer after the rows "
            "have run; an absent one means not-yet-produced, wrong root, or deleted, and "
            "never that the per-row roles do not apply",
            ref=ref,
            custody_ref=locator.ref,
        ) from exc
    except (OSError, ValueError) as exc:
        raise RowCustodyFulfillmentError(
            f"binds per-row custody from {locator.ref!r}, which does not load as a "
            f"RowIndexCustodyBindings document: {exc}",
            ref=ref,
            custody_ref=locator.ref,
        ) from exc
    if bindings.index_id != locator.index_id:
        raise RowCustodyFulfillmentError(
            f"binds per-row custody from {locator.ref!r}, whose bindings belong to row index "
            f"{bindings.index_id!r} rather than {locator.index_id!r}",
            ref=ref,
            custody_ref=locator.ref,
        )
    return bindings, path


def _resolve_inputs(
    record: RowExpansionRecord,
    bindings: RowIndexCustodyBindings,
    *,
    ref: str,
    custody_ref: str,
):
    """Expand this figure's roles over its rows against located custody."""
    try:
        resolved = resolve_figure_input_roles(record.request, record.resolved_rows, bindings)
    except (RowSelectionError, FigureRoleReferenceError) as exc:
        raise RowCustodyFulfillmentError(
            f"binds per-row custody from {custody_ref!r}, which does not satisfy this "
            f"figure's declared roles over its expanded rows: {exc}",
            ref=ref,
            custody_ref=custody_ref,
        ) from exc
    if not resolved.fully_bound:
        raise RowCustodyFulfillmentError(
            f"binds per-row custody from {custody_ref!r} and is still awaiting a run receipt "
            f"for {list(resolved.pending_roles)}; a located custody document that leaves a "
            "declared role pending is incomplete for this figure",
            ref=ref,
            custody_ref=custody_ref,
        )
    return resolved


__all__ = [
    "RESOLVED_ROW_SET_CONTRIBUTION",
    "ROW_CUSTODY_CONTRIBUTION",
    "ROW_EXPANSION_CONTRIBUTION",
    "RowCustodyFulfillmentError",
    "RowCustodyOverlay",
    "RowExpansionRecord",
    "read_row_expansion_record",
    "resolve_row_custody_overlay",
]
