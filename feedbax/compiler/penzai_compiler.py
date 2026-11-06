"""Authoring-time compiler diagnostics for Penzai adapter interiors."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from typing import Any

from jaxtyping import PyTree

from feedbax.components.penzai import (
    PENZAI_AVAILABLE,
    TREESCOPE_AVAILABLE,
    build_penzai_subgraph,
    UnknownPenzaiBuilderError,
    list_penzai_builder_info,
    penzai_state_variable_paths,
    penzai_structure_summary,
    render_penzai_model_html,
    resolve_penzai_model,
)
from feedbax.contracts.domain import (
    DomainCompileReport,
    DomainDiagnostic,
    report_status_for_diagnostics,
)

PENZAI_COMPILER_ID = "feedbax.compiler.penzai"


def penzai_builder_options() -> list[dict[str, Any]]:
    """Return registry-visible Penzai builder metadata."""

    return list_penzai_builder_info()


def compile_penzai_authoring_report(
    *,
    builder_name: str,
    params: Mapping[str, Any] | None = None,
    input_port: str = "input",
    output_port: str = "output",
    node_path: list[str] | None = None,
) -> DomainCompileReport:
    """Return a structural Penzai authoring report without training execution."""

    report, _ = _analyze_penzai_model(
        builder_name=builder_name,
        params=params,
        input_port=input_port,
        output_port=output_port,
        node_path=node_path,
    )
    return report


def inspect_penzai_model(
    *,
    builder_name: str,
    params: Mapping[str, Any] | None = None,
    input_port: str = "input",
    output_port: str = "output",
    node_path: list[str] | None = None,
) -> tuple[DomainCompileReport, str]:
    """Inspect and render the same model in one operation."""
    report, model = _analyze_penzai_model(
        builder_name=builder_name,
        params=params,
        input_port=input_port,
        output_port=output_port,
        node_path=node_path,
    )
    return report, "" if report.status == "error" else render_penzai_model_html(model)


def _analyze_penzai_model(
    *,
    builder_name: str,
    params: Mapping[str, Any] | None,
    input_port: str,
    output_port: str,
    node_path: list[str] | None,
) -> tuple[DomainCompileReport, PyTree]:
    node_ids = list(node_path or [])
    payload = {
        "builder_name": builder_name,
        "params": dict(params or {}),
        "input_port": input_port,
        "output_port": output_port,
    }
    content_hash = _stable_hash(payload)
    diagnostics: list[DomainDiagnostic] = []
    summary: dict[str, int] = {}
    derived_interface = {
        "inputs": {input_port: {"dtype": "any", "kind": "signal"}},
        "outputs": {output_port: {"dtype": "any", "kind": "signal"}},
    }

    if not PENZAI_AVAILABLE:
        diagnostics.append(
            DomainDiagnostic(
                severity="error",
                code="penzai.unavailable",
                message="Penzai is not installed; this adapter cannot be inspected or compiled.",
                node_ids=node_ids,
            )
        )
        return _report(content_hash, diagnostics, summary, derived_interface), None

    try:
        model = resolve_penzai_model(builder_name, params)
        state_paths = penzai_state_variable_paths(model)
        summary = penzai_structure_summary(model)
    except UnknownPenzaiBuilderError:
        diagnostics.append(
            DomainDiagnostic(
                severity="error",
                code="penzai.unknown_builder",
                message=(
                    f"Unknown Penzai builder {builder_name!r}. "
                    f"Available builders: {[item['name'] for item in penzai_builder_options()]}"
                ),
                node_ids=node_ids,
                details={"builder_name": builder_name},
            )
        )
        return _report(content_hash, diagnostics, summary, derived_interface), None
    except Exception as exc:  # noqa: BLE001 - authoring endpoint reports, not raises.
        diagnostics.append(
            DomainDiagnostic(
                severity="error",
                code="penzai.builder_failed",
                message=f"Penzai builder {builder_name!r} failed during structural compile: {exc}",
                node_ids=node_ids,
                details={"builder_name": builder_name},
            )
        )
        return _report(content_hash, diagnostics, summary, derived_interface), None

    if state_paths:
        diagnostics.append(
            DomainDiagnostic(
                severity="error",
                code="penzai.stateful_unsupported",
                message=(
                    "PenzaiAdapter does not support stateful Penzai models yet. "
                    f"StateVariable leaves: {', '.join(state_paths)}."
                ),
                node_ids=node_ids,
                variables=state_paths,
                counts={"state_variables": len(state_paths)},
                details={"builder_name": builder_name},
            )
        )

    return _report(content_hash, diagnostics, summary, derived_interface), model


def render_penzai_treescope_html(
    *,
    builder_name: str,
    params: Mapping[str, Any] | None = None,
    input_port: str = "input",
    output_port: str = "output",
) -> str:
    """Render a Penzai adapter model as Treescope HTML."""

    if not TREESCOPE_AVAILABLE:
        raise ImportError("treescope is required for Penzai inspector rendering.")
    component = build_penzai_subgraph(
        builder_name=builder_name,
        params=params,
        input_port=input_port,
        output_port=output_port,
    )
    return component.treescope_html()


def _report(
    content_hash: str,
    diagnostics: list[DomainDiagnostic],
    summary: dict[str, int],
    derived_interface: dict[str, Any],
) -> DomainCompileReport:
    return DomainCompileReport(
        status=report_status_for_diagnostics(diagnostics),
        interior_content_hash=content_hash,
        diagnostics=diagnostics,
        derived_interface=derived_interface,
        summary=summary,
    )


def _stable_hash(payload: Mapping[str, Any]) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str).encode()
    return hashlib.sha256(encoded).hexdigest()
