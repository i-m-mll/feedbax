"""Cold-start conformance fixture: one fake project, one data declaration.

This package stands in for a real downstream project while the engine is being
upstreamed. Its vocabulary is deliberately invented — ``quillon`` names nothing
and belongs to nobody — so the fixture proves the contract rather than any
particular science.

What it proves is how little a project now says, and how it says it. The
declaration is a six-key JSON document at the project root: a schema identity, a
name, two directories, and one repo-relative budget path. It is *data*, read
directly by :func:`~feedbax.contracts.project_experiment.load_project_declaration`
from a stated root — the fixture registers no plugin to announce it, because a
layout fact is not an implementation. There is no envelope family to claim, no
layer to bind, no lowerer, no applicability callback, and no compiler contract,
because Feedbax owns the one dialect and the one compiler for it. Everything
``quillon`` contributes to a compiled document, it contributes as *data inside*
native Feedbax composition deltas: dotted paths, values, recipe ids, and
input-role strings. What ``quillon`` does register through the ordinary plugin
bootstrap is genuine science — its recipes, in :mod:`tests.fulfillment_cli_plugin`.

:data:`PROJECT_DECLARATION_DOCUMENT` is the single source of truth for the
declaration. :data:`PROJECT_DECLARATION` is that same document parsed against
this package's own resources, for tests that drive the kernel without laying out
a repository; :func:`write_repo` writes it, its budget document, five frozen
bases, and five envelopes into one real repository root, which is what the
kernel, dialect, and entrypoint tests compile against.
"""

from __future__ import annotations

import json
from importlib import resources
from pathlib import Path
from typing import Any

from feedbax.contracts.project_experiment import (
    PROJECT_DECLARATION_FILENAME,
    PROJECT_EXPERIMENT_DECLARATION_SCHEMA_ID,
    PROJECT_EXPERIMENT_DECLARATION_SCHEMA_VERSION,
    parse_project_declaration,
)

PROJECT = "quillon"
DECLARATION_SOURCE = f"tests.fake_project_experiment:{PROJECT_DECLARATION_FILENAME}"

ENVELOPE_DIRECTORY = "studies"
OUTPUT_DIRECTORY = "compiled"
BASE_DIRECTORY = "bases"
BUDGET_REF = "budgets/quillon.envelope_budgets.json"

PROJECT_DECLARATION_DOCUMENT: dict[str, Any] = {
    "schema_id": PROJECT_EXPERIMENT_DECLARATION_SCHEMA_ID,
    "schema_version": PROJECT_EXPERIMENT_DECLARATION_SCHEMA_VERSION,
    "project": PROJECT,
    "envelope_directory": ENVELOPE_DIRECTORY,
    "output_directory": OUTPUT_DIRECTORY,
    "authoring_budget": BUDGET_REF,
}

#: The declaration document's exact tracked bytes, as a project root holds them.
PROJECT_DECLARATION_BYTES = (
    json.dumps(PROJECT_DECLARATION_DOCUMENT, indent=2) + "\n"
).encode("utf-8")

#: The budget document ``quillon`` authors, kept as package data so a kernel test
#: can load budgets without first writing a repository.
BUDGET_BYTES = resources.files(__name__).joinpath(*BUDGET_REF.split("/")).read_bytes()

PROJECT_DECLARATION = parse_project_declaration(
    PROJECT_DECLARATION_BYTES,
    budget_root=resources.files(__name__),
    source=DECLARATION_SOURCE,
)


# -- repository scaffolding used by the kernel and dialect tests ----------

SURVEY_PAYLOAD: dict[str, Any] = {
    "schema_id": "quillon.survey_payload",
    "span": 4,
    "cadence": 2,
    "probe": {"depth": 1},
}

TRAINING_BASE = f"{BASE_DIRECTORY}/baseline.training_run_matrix.json"
EVALUATION_BASE = f"{BASE_DIRECTORY}/baseline.evaluation_run_matrix.json"
ANALYSIS_BASE = f"{BASE_DIRECTORY}/baseline.analysis_run.json"
FIGURE_BASE = f"{BASE_DIRECTORY}/baseline.figure.json"
REPORT_BASE = f"{BASE_DIRECTORY}/baseline.report.json"

BASE_DOCUMENTS: dict[str, dict[str, Any]] = {
    TRAINING_BASE: {
        "schema_id": "feedbax.spec.training_run_matrix",
        "schema_version": "feedbax.spec.training_run_matrix.v5",
        "name": "baseline",
        "base": {"kind": "inline", "inline": SURVEY_PAYLOAD},
        "rows": [{"row_id": "baseline", "seed": 42, "overrides": []}],
        "tags": ["baseline"],
    },
    EVALUATION_BASE: {
        "schema_id": "feedbax.spec.evaluation_run_matrix",
        "schema_version": "feedbax.spec.evaluation_run_matrix.v3",
        "base": {
            "schema_id": "feedbax.spec.evaluation_run",
            "schema_version": "feedbax.spec.evaluation_run.v1",
            "evaluation_type": "quillon.span_probe",
            "params": {"cadence": 2},
        },
        "rows": [{"row_id": "baseline"}],
    },
    ANALYSIS_BASE: {
        "schema_id": "feedbax.spec.analysis_run",
        "schema_version": "feedbax.spec.analysis_run.v2",
        "analysis_type": "quillon.span_summary",
        "params": {"window": 3},
    },
    FIGURE_BASE: {
        "schema_id": "feedbax.spec.figure",
        "schema_version": "feedbax.spec.figure.v2",
        "name": "baseline-span",
        "assembler": "quillon.span_assembler",
        "assembler_params": {},
    },
    REPORT_BASE: {
        "schema_id": "feedbax.spec.report.ordered_figure",
        "schema_version": "feedbax.spec.report.ordered_figure.v3",
        "title": "Quillon baseline span",
        "sections": [{"title": "Span", "figures": [], "tables": []}],
    },
}

TRAINING_ENVELOPE: dict[str, Any] = {
    "schema": "feedbax.experiment_envelope.v1",
    "name": "widened",
    "base": TRAINING_BASE,
    "issue": "q1a2b3c",
    "reason": "The baseline span saturates the probe before the cadence window closes.",
    "assert": [{"path": "base.inline.cadence", "equals": 2}],
    "training": {
        "rows": [
            {
                "from": "baseline",
                "id": "widened",
                "seed": 43,
                "replaces": {"row": "baseline", "seed": 42},
                "delta": {
                    "layer_id": "widened-span",
                    "patches": [
                        {"path": "span", "op": "replace", "value": 9},
                        {"path": "probe.depth", "op": "remove"},
                        {
                            "path": "cadence_profile",
                            "op": "add",
                            "value": {"steps": [1, 2], "hold": 3},
                        },
                    ],
                },
            }
        ],
        "tags": {"add": ["widened"], "remove": ["baseline"]},
        "checkpoint_initialization": [
            {
                "row": "widened",
                "mode": "continue_from",
                "source": {
                    "kind": "receipt",
                    "manifest_kind": "quillon.survey_run",
                    "manifest_id": "baseline-0",
                    "manifest_sha256": "1f0e3dad99908345f7439f8ffabdffc4e0e0e0e0e0e0e0e0e0e0e0e0e0e0e0e0",
                    "size_bytes": 4096,
                    "execution_uri": "file:///custody/quillon/baseline-0"
                }
            }
        ]
    },
}

EVALUATION_ENVELOPE: dict[str, Any] = {
    "schema": "feedbax.experiment_envelope.v1",
    "name": "widened-probe",
    "base": EVALUATION_BASE,
    "evaluation": {
        "subject": {"kind": "envelope", "alias": "widened"},
        "subject_id": "widened",
        "params": {"depth": 3},
    },
}

ANALYSIS_ENVELOPE: dict[str, Any] = {
    "schema": "feedbax.experiment_envelope.v1",
    "name": "widened-summary",
    "base": ANALYSIS_BASE,
    "analysis": {
        "target": "run",
        "subjects": [
            {
                "alias": "probe",
                "role": "observations",
                "ref": {
                    "kind": "receipt",
                    "manifest_kind": "quillon.probe_run",
                    "manifest_id": "widened-probe-0",
                },
            }
        ],
        "params": {"trim": 1},
    },
}

FIGURE_ENVELOPE: dict[str, Any] = {
    "schema": "feedbax.experiment_envelope.v1",
    "name": "widened-plot",
    "base": FIGURE_BASE,
    "figure": {
        "inputs": [
            {
                "input_role": "observed",
                "ref": {"kind": "envelope", "alias": "widened-summary"},
            }
        ],
        "delta": {
            "layer_id": "widened-plot",
            "patches": [
                {"path": "assembler_params.style", "op": "add", "value": "wide"}
            ],
        },
    },
}

REPORT_ENVELOPE: dict[str, Any] = {
    "schema": "feedbax.experiment_envelope.v1",
    "name": "widened-report",
    "base": REPORT_BASE,
    "report": {
        "bindings": [
            {
                "role_path": "sections.0.figures.0",
                "ref": {"kind": "envelope", "alias": "widened-plot"},
            },
            {
                "role_path": "sections.0.tables.0",
                "ref": {
                    "kind": "not_applicable",
                    "reason": "the widened survey has no comparison arm to tabulate",
                },
            },
        ],
        "delta": {
            "layer_id": "widened-report",
            "patches": [
                {"path": "title", "op": "replace", "value": "Quillon widened span"}
            ],
        },
    },
}

ENVELOPES: dict[str, dict[str, Any]] = {
    "widened": TRAINING_ENVELOPE,
    "widened-probe": EVALUATION_ENVELOPE,
    "widened-summary": ANALYSIS_ENVELOPE,
    "widened-plot": FIGURE_ENVELOPE,
    "widened-report": REPORT_ENVELOPE,
}


def write_json(path: Path, document: Any) -> None:
    """Write one JSON document in the readable tracked form."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(document, indent=2) + "\n", encoding="utf-8")


def write_envelope(path: Path, document: Any) -> None:
    """Write one authored envelope in a form the pre-parse guards admit."""
    write_json(path, document)


def envelope_path(root: Path, alias: str) -> Path:
    """Return where one alias's authored envelope lives under *root*."""
    return root / ENVELOPE_DIRECTORY / f"{alias}.envelope.json"


def write_declaration(root: Path) -> Path:
    """Write the project declaration and its budget document into *root*."""
    declaration = root / PROJECT_DECLARATION_FILENAME
    declaration.parent.mkdir(parents=True, exist_ok=True)
    declaration.write_bytes(PROJECT_DECLARATION_BYTES)
    budget = root / BUDGET_REF
    budget.parent.mkdir(parents=True, exist_ok=True)
    budget.write_bytes(BUDGET_BYTES)
    return declaration


def write_repo(root: Path, *, envelopes: dict[str, dict[str, Any]] | None = None) -> None:
    """Lay out one quillon repository: a declaration, five bases, five envelopes."""
    write_declaration(root)
    for ref, document in BASE_DOCUMENTS.items():
        write_json(root / ref, document)
    for alias, document in (ENVELOPES if envelopes is None else envelopes).items():
        write_envelope(envelope_path(root, alias), document)


__all__ = [
    "ANALYSIS_BASE",
    "ANALYSIS_ENVELOPE",
    "BASE_DOCUMENTS",
    "BUDGET_BYTES",
    "BUDGET_REF",
    "DECLARATION_SOURCE",
    "ENVELOPES",
    "ENVELOPE_DIRECTORY",
    "EVALUATION_BASE",
    "EVALUATION_ENVELOPE",
    "FIGURE_BASE",
    "FIGURE_ENVELOPE",
    "OUTPUT_DIRECTORY",
    "PROJECT",
    "PROJECT_DECLARATION",
    "PROJECT_DECLARATION_BYTES",
    "PROJECT_DECLARATION_DOCUMENT",
    "REPORT_BASE",
    "REPORT_ENVELOPE",
    "SURVEY_PAYLOAD",
    "TRAINING_BASE",
    "TRAINING_ENVELOPE",
    "envelope_path",
    "write_declaration",
    "write_envelope",
    "write_json",
    "write_repo",
]
