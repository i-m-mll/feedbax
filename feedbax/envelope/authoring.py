"""Reading an authored document: noncanonical bytes and over-budget content.

Formatting, duplicate keys, and oversized input are rejected *before* JSON
parsing, so an ambiguous or oversized document can never reach a schema layer and
be silently normalized there. Which layer's caps apply is a property of the
document, so the pre-parse byte guard uses the *widest* cap any layer states, and
the document's own layer cap is applied the moment its layer is known — still
before any project model sees it.

The layer is identified by the project, not guessed at by the engine: the caller
passes the label site its declaration reads, and a document whose layer cannot be
determined stays under the widest caps. Admitting it there is not a loophole,
because the project's own closed authored schema still refuses a document that
authors no layer or more than one.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Mapping
from typing import Any, NoReturn

from feedbax.contracts.strict_json import DuplicateJsonKeyError, strict_json_loads

from feedbax.contracts.authored_canonical import emit_text
from feedbax.contracts.authoring_budget import AuthoringBudgets, LayerBudget
from feedbax.contracts.experiment_envelope import (
    ExperimentEnvelopeRejection,
    ExperimentEnvelopeRejectionCategory,
)

_PROSE_HOME = (
    "an envelope carries the scientific delta only; everything else is inherited "
    "from its base, derived by the compiler, or belongs in the run receipt"
)


def _reject(
    category: ExperimentEnvelopeRejectionCategory,
    field: str,
    message: str,
    *,
    correct_home: str | None = None,
) -> NoReturn:
    raise ExperimentEnvelopeRejection(category, message, field=field, correct_home=correct_home)


def _reject_constant(name: str) -> NoReturn:
    _reject(
        ExperimentEnvelopeRejectionCategory.NONCANONICAL_FORMAT,
        name,
        "non-finite JSON constants are not admitted",
    )


def _walk_budget(value: Any, budget: LayerBudget, field: str, depth: int) -> int:
    """Count syntax nodes, refusing the first structural cap the document exceeds."""
    if depth > budget.max_depth:
        _reject(
            ExperimentEnvelopeRejectionCategory.BUDGET_EXCEEDED,
            field,
            f"nesting depth exceeds {budget.cap('max_depth')}",
        )
    if isinstance(value, dict):
        if len(value) > budget.max_object_keys:
            _reject(
                ExperimentEnvelopeRejectionCategory.BUDGET_EXCEEDED,
                field,
                f"{len(value)} keys exceeds {budget.cap('max_object_keys')}",
            )
        return 1 + sum(
            1 + _walk_budget(item, budget, f"{field}.{key}", depth + 1)
            for key, item in value.items()
        )
    if isinstance(value, list):
        if len(value) > budget.max_items:
            _reject(
                ExperimentEnvelopeRejectionCategory.BUDGET_EXCEEDED,
                field,
                f"{len(value)} items exceeds {budget.cap('max_items')}",
            )
        return 1 + sum(
            _walk_budget(item, budget, f"{field}[{index}]", depth + 1)
            for index, item in enumerate(value)
        )
    if isinstance(value, str):
        encoded = len(value.encode("utf-8"))
        if encoded > budget.max_scalar_bytes:
            _reject(
                ExperimentEnvelopeRejectionCategory.BUDGET_EXCEEDED,
                field,
                f"scalar of {encoded} bytes exceeds {budget.cap('max_scalar_bytes')} bytes",
                correct_home="prose belongs on the change's tracking issue, not in the envelope",
            )
    return 1


def guard_authored_bytes(raw: bytes, budget: LayerBudget, *, field: str) -> str:
    """Refuse noncanonical or oversized authored bytes before they are parsed."""
    if raw.startswith(b"\xef\xbb\xbf"):
        _reject(
            ExperimentEnvelopeRejectionCategory.NONCANONICAL_FORMAT,
            field,
            "byte-order mark is not admitted",
        )
    if len(raw) > budget.max_bytes:
        _reject(
            ExperimentEnvelopeRejectionCategory.BUDGET_EXCEEDED,
            field,
            f"{len(raw)} bytes exceeds {budget.cap('max_bytes')}",
        )
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise ExperimentEnvelopeRejection(
            ExperimentEnvelopeRejectionCategory.NONCANONICAL_FORMAT,
            f"not valid UTF-8: {exc}",
            field=field,
        ) from exc
    if "\r" in text or "\t" in text:
        _reject(
            ExperimentEnvelopeRejectionCategory.NONCANONICAL_FORMAT,
            field,
            "carriage returns and tabs are not admitted",
        )
    if text.endswith("\n\n"):
        _reject(
            ExperimentEnvelopeRejectionCategory.NONCANONICAL_FORMAT,
            field,
            "the document must end with at most one newline",
        )
    for number, line in enumerate(text.rstrip("\n").split("\n"), start=1):
        if line != line.rstrip():
            _reject(
                ExperimentEnvelopeRejectionCategory.NONCANONICAL_FORMAT,
                f"{field}:{number}",
                "trailing whitespace is not admitted",
            )
    return text


def enforce_budget(document: Any, budget: LayerBudget, raw_length: int, *, field: str) -> None:
    """Apply one layer's caps to a parsed authored document."""
    if raw_length > budget.max_bytes:
        _reject(
            ExperimentEnvelopeRejectionCategory.BUDGET_EXCEEDED,
            field,
            f"{raw_length} bytes exceeds {budget.cap('max_bytes')}",
        )
    nodes = _walk_budget(document, budget, field, 1)
    if nodes > budget.max_ast_nodes:
        _reject(
            ExperimentEnvelopeRejectionCategory.BUDGET_EXCEEDED,
            field,
            f"{nodes} syntax nodes exceeds {budget.cap('max_ast_nodes')}",
            correct_home=_PROSE_HOME,
        )
    lines = len(emit_text(document).splitlines())
    if lines > budget.max_lines:
        _reject(
            ExperimentEnvelopeRejectionCategory.BUDGET_EXCEEDED,
            field,
            f"{lines} canonically formatted lines exceeds {budget.cap('max_lines')}",
            correct_home=_PROSE_HOME,
        )


def enforce_assertion_budget(count: int, budget: LayerBudget, *, field: str) -> None:
    """Refuse an envelope that states more preconditions than its layer allows."""
    if count > budget.max_assertions:
        _reject(
            ExperimentEnvelopeRejectionCategory.BUDGET_EXCEEDED,
            field,
            f"{count} assertions exceeds {budget.cap('max_assertions')}",
            correct_home="an assertion guards a precondition the change depends on; a survey "
            "of the base's contents is not one",
        )


def enforce_row_budget(count: int, budget: LayerBudget, *, field: str) -> None:
    """Refuse a layer that authors more rows than its budget admits.

    ``max_rows`` is optional because only the training layer authors rows: a
    layer whose budget states no ``max_rows`` bounds nothing here. The count is
    the rows the *envelope* authors, not the rows the compiled matrix ends up
    with, so a refusal is always fixable by editing the envelope rather than by
    editing whatever the envelope inherits from.
    """
    cap = budget.optional_cap("max_rows")
    if cap is not None and count > cap:
        _reject(
            ExperimentEnvelopeRejectionCategory.BUDGET_EXCEEDED,
            field,
            f"{count} authored rows exceeds {budget.optional_cap_label('max_rows')}",
            correct_home="a row is one run of the experiment; a sweep wider than the budget "
            "is a separate experiment, authored as its own envelope",
        )


def read_authored_document(
    raw: bytes,
    budgets: AuthoringBudgets,
    *,
    field: str,
    layer_of: Callable[[Mapping[str, Any]], str | None] | None = None,
) -> dict[str, Any]:
    """Parse one authored document under its layer's budget.

    Args:
        raw: The authored bytes exactly as tracked.
        budgets: The project's per-layer budget policy.
        field: The locator used in a rejection, normally the document's path.
        layer_of: How the project reads which layer a parsed document authors.
            A document whose layer is undeterminable stays under the widest caps.

    Returns:
        The parsed document, already proven to be within its layer's budget.

    Raises:
        ExperimentEnvelopeRejection: The bytes are noncanonical, the document is
            ambiguous, or it exceeds a cap its layer states.
    """
    widest = budgets.widest
    text = guard_authored_bytes(raw, widest, field=field)
    try:
        document = strict_json_loads(text, ref=field, parse_constant=_reject_constant)
    except DuplicateJsonKeyError as exc:
        raise ExperimentEnvelopeRejection(
            ExperimentEnvelopeRejectionCategory.DUPLICATE_KEY,
            "the same key is authored twice in one object",
            field=exc.key,
        ) from exc
    except ExperimentEnvelopeRejection:
        raise
    except json.JSONDecodeError as exc:
        raise ExperimentEnvelopeRejection(
            ExperimentEnvelopeRejectionCategory.NONCANONICAL_FORMAT,
            f"invalid JSON: {exc.msg}",
            field=f"{field}:{exc.lineno}",
        ) from exc
    if not isinstance(document, dict):
        _reject(
            ExperimentEnvelopeRejectionCategory.INVALID_VALUE,
            field,
            "an authored document is a JSON object",
        )
    layer = layer_of(document) if layer_of is not None else None
    budget = widest if layer is None else budgets.for_layer(layer)
    enforce_budget(document, budget, len(raw), field=field)
    return document


__all__ = [
    "enforce_assertion_budget",
    "enforce_budget",
    "enforce_row_budget",
    "guard_authored_bytes",
    "read_authored_document",
]
