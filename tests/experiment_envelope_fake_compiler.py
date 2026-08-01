"""Minimal downstream envelope compiler proving the dispatch contract.

This module stands in for a real downstream compiler while one is being built
in its own repository. It exists to prove exactly one thing: that a compiler
registered through the ordinary plugin bootstrap, with its registry injected,
is found by ``python -m feedbax preflight-experiment-envelope`` purely from the
envelope's declared schema string — no import-time discovery, no module-path
convention, and no feedbax knowledge of the dialect.

Its dialect is deliberately trivial: it accepts a ``name``, writes a compile
lock and one document, and rejects an envelope carrying ``metadata``.
"""

from __future__ import annotations

import json

from feedbax.plugins import (
    EXPERIMENT_ENVELOPE_COMPILERS,
    ExperimentEnvelopeCompileRequest,
    ExperimentEnvelopeCompileResult,
    ExperimentEnvelopeCompilerRegistration,
    ExperimentEnvelopeRejection,
    ExperimentEnvelopeRejectionCategory,
    FamilyRequirement,
    PluginDeclaration,
    PluginRegistration,
)

ENVELOPE_SCHEMA = "tests.fake_experiment.v1"
FAMILY = "fake_document"


def compile_fake_envelope(
    request: ExperimentEnvelopeCompileRequest,
) -> ExperimentEnvelopeCompileResult:
    """Compile the fake dialect, or reject it with an actionable diagnostic."""
    envelope = request.envelope
    if "metadata" in envelope:
        raise ExperimentEnvelopeRejection(
            ExperimentEnvelopeRejectionCategory.UNKNOWN_FIELD,
            "the fake experiment envelope admits no free-form metadata",
            field="metadata",
            correct_home="the issue this experiment is filed under",
        )
    name = envelope.get("name")
    if not isinstance(name, str) or not name:
        raise ExperimentEnvelopeRejection(
            ExperimentEnvelopeRejectionCategory.UNKNOWN_FIELD,
            "the fake experiment envelope requires a nonempty name",
            field="name",
            correct_home="the envelope's identity block",
        )
    request.out_dir.mkdir(parents=True, exist_ok=True)
    lock_name = f"{name}.compile-lock.json"
    document_name = f"{name}.{FAMILY}.json"
    (request.out_dir / lock_name).write_text(
        json.dumps({"envelope_schema": ENVELOPE_SCHEMA, "name": name}, sort_keys=True),
        encoding="utf-8",
    )
    (request.out_dir / document_name).write_text(
        json.dumps({"name": name}, sort_keys=True), encoding="utf-8"
    )
    return ExperimentEnvelopeCompileResult(
        envelope_schema=ENVELOPE_SCHEMA,
        name=name,
        family=FAMILY,
        compile_lock_path=lock_name,
        document_path=document_name,
    )


def _register(context) -> None:
    context.registry(EXPERIMENT_ENVELOPE_COMPILERS).register(
        ExperimentEnvelopeCompilerRegistration(
            envelope_schema=ENVELOPE_SCHEMA,
            owner="tests.experiment_envelope_fake_compiler",
            compile=compile_fake_envelope,
        )
    )


PLUGIN_REGISTRATION = PluginRegistration(
    PluginDeclaration(
        "tests.experiment_envelope_fake_compiler",
        "1.0",
        1,
        families=(FamilyRequirement("experiment_envelope_compilers"),),
    ),
    _register,
)
