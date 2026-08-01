"""The fake project, wired to the generic engine kernel.

Lane A1's :mod:`tests.fake_project_extension` proves a project can *declare* what
it extends. This module proves the declaration is enough to *drive* the engine:
the kernel resolves parents, lineage, assertions, echoes, budgets, and locks,
and ``quillon`` supplies only the two things the engine cannot know — which layer
a document belongs to, and how a layer compiles.

The vocabulary stays invented. ``quillon`` names nothing, a ``survey`` states a
settings delta over a parent survey, and a ``digest`` states prose about one
survey it names by alias. Two layers is the smallest shape that exercises both a
same-layer parent and a cross-layer reference, which is what the kernel's two
resolution paths need.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from copy import deepcopy
from pathlib import Path
from typing import Any

from feedbax.contracts.authoring_budget import AuthoringBudgets, load_authoring_budget_document
from feedbax.contracts.experiment_compile_lock import (
    CompilerImplementation,
    PlannedProductReference,
    ReportParentBinding,
)
from feedbax.contracts.experiment_envelope import (
    ExperimentEnvelopeRejection,
    ExperimentEnvelopeRejectionCategory,
)
from feedbax.envelope import (
    AuthoredAssertion,
    CompiledLayer,
    EnvelopeKernel,
    EnvelopeLayout,
    LayerCompileContext,
    ProjectCompilerHooks,
    check_echo,
)

from tests.fake_project_extension import PROJECT_DECLARATION

SURVEY_FAMILY = "quillon.survey_document"
DIGEST_FAMILY = "quillon.digest_document"

#: Which layer each compiled or frozen document belongs to. A document that
#: declares none of these belongs to no quillon layer, which is what makes a
#: cross-layer base a rejection rather than a silent acceptance.
FAMILY_LAYERS: Mapping[str, str] = {
    SURVEY_FAMILY: "survey",
    DIGEST_FAMILY: "digest",
}

QUILLON_LAYOUT = EnvelopeLayout(
    envelope_directory="studies",
    output_directory="compiled",
)

QUILLON_IMPLEMENTATION = CompilerImplementation(
    code_unit="tests.fake_project_extension.kernel",
    packages=("feedbax",),
)


def layer_of(document: Mapping[str, Any]) -> str | None:
    """Return the quillon layer a document belongs to, or ``None``.

    An authored envelope says so directly in its ``layer`` field. A compiled or
    frozen document says so through the family it declares.
    """
    authored = document.get("layer")
    if isinstance(authored, str) and authored in PROJECT_DECLARATION.labels("layer"):
        return authored
    return FAMILY_LAYERS.get(str(document.get("schema_id")))


def authored_assertions(envelope: Mapping[str, Any]) -> tuple[AuthoredAssertion, ...]:
    """Read the inherited preconditions an envelope states."""
    stated = envelope.get("assert") or []
    if not isinstance(stated, list):
        raise ExperimentEnvelopeRejection(
            ExperimentEnvelopeRejectionCategory.INVALID_VALUE,
            "'assert' is a list of {path, equals} objects",
            field="assert",
        )
    return tuple(
        AuthoredAssertion(path=str(item["path"]), equals=item["equals"]) for item in stated
    )


def _require_name(envelope: Mapping[str, Any]) -> str:
    name = envelope.get("name")
    if not isinstance(name, str) or not name:
        raise ExperimentEnvelopeRejection(
            ExperimentEnvelopeRejectionCategory.MISSING_FIELD,
            "a quillon study envelope requires a nonempty name",
            field="name",
            correct_home="the envelope's identity block",
        )
    return name


def _compile_survey(context: LayerCompileContext) -> CompiledLayer:
    """Compile one settings delta over a parent survey document."""
    envelope = context.envelope
    parent = context.parent
    if parent is None:
        raise ExperimentEnvelopeRejection(
            ExperimentEnvelopeRejectionCategory.UNRESOLVED_BASE,
            "a survey envelope states the survey it changes",
            field="base",
        )
    name = _require_name(envelope)
    body = envelope.get("body") or {}
    settings = body.get("settings") or {}

    if name == parent.pinned.document.get("name"):
        check_echo(context.lineage, "name", name, field="envelope.name")
    for key in sorted(settings):
        check_echo(
            context.lineage, f"settings.{key}", settings[key], field=f"body.settings.{key}"
        )

    inherited = dict(parent.pinned.document.get("settings") or {})
    compiled = {
        key: deepcopy(value)
        for key, value in parent.pinned.document.items()
        if key not in ("metadata", "base")
    }
    compiled["name"] = name
    compiled["base"] = parent.base_block()
    compiled["settings"] = {**inherited, **settings}

    return CompiledLayer(
        name=name,
        family=SURVEY_FAMILY,
        document=compiled,
        resolved_deltas={"survey_settings": dict(settings)},
        overridden_paths={
            "name": "envelope.name",
            **{f"settings.{key}": f"body.settings.{key}" for key in settings},
        },
        issue=envelope.get("issue"),
    )


def _compile_digest(context: LayerCompileContext) -> CompiledLayer:
    """Compile one prose digest that names the survey it describes."""
    envelope = context.envelope
    parent = context.parent
    if parent is None:
        raise ExperimentEnvelopeRejection(
            ExperimentEnvelopeRejectionCategory.UNRESOLVED_BASE,
            "a digest envelope states the digest it changes",
            field="base",
        )
    name = _require_name(envelope)
    body = envelope.get("body") or {}
    subject_alias = body.get("of")
    if not isinstance(subject_alias, str) or not subject_alias:
        raise ExperimentEnvelopeRejection(
            ExperimentEnvelopeRejectionCategory.MISSING_FIELD,
            "a digest names the survey envelope it describes",
            field="body.of",
        )
    summary = body.get("summary")
    check_echo(context.lineage, "summary", summary, field="body.summary")

    subject = context.upstream_pin(subject_alias, field="body.of")
    compiled = {
        key: deepcopy(value)
        for key, value in parent.pinned.document.items()
        if key not in ("metadata", "base")
    }
    compiled["name"] = name
    compiled["base"] = parent.base_block()
    compiled["summary"] = summary
    compiled["subject"] = subject

    return CompiledLayer(
        name=name,
        family=DIGEST_FAMILY,
        document=compiled,
        resolved_deltas={"digest_summary": summary},
        references=[
            PlannedProductReference(
                envelope_ref=subject["envelope"]["ref"],
                envelope_hash=subject["envelope"]["envelope_hash"],
                product_name=subject["name"],
                product_schema_id=SURVEY_FAMILY,
                product_schema_version=f"{SURVEY_FAMILY}.v1",
                compiled_content_hash=subject["compiled_document"]["content_hash"],
                role_path="body.of",
                consumer=ReportParentBinding(
                    parent_kind=SURVEY_FAMILY, parent_id=subject_alias
                ),
            )
        ],
        identity_contributions={"subject": subject},
        overridden_paths={"name": "envelope.name", "summary": "body.summary"},
        issue=envelope.get("issue"),
    )


_LAYER_COMPILERS = {"survey": _compile_survey, "digest": _compile_digest}


def compile_layer(context: LayerCompileContext) -> CompiledLayer:
    """Dispatch to the compiler for the layer the engine already resolved."""
    return _LAYER_COMPILERS[context.layer](context)


def validate_compiled(document: Any, family: str, envelope_ref: str) -> None:
    """Prove a compiled document really is a member of the family it claims."""
    declared = document.get("schema_id") if isinstance(document, Mapping) else None
    if declared != family:
        raise ExperimentEnvelopeRejection(
            ExperimentEnvelopeRejectionCategory.INVALID_VALUE,
            f"the compiled document declares {declared!r} but was compiled as {family!r}",
            field=envelope_ref,
        )


QUILLON_HOOKS = ProjectCompilerHooks(
    layer_of=layer_of,
    compile_layer=compile_layer,
    authored_assertions=authored_assertions,
    validate_compiled=validate_compiled,
)


def load_quillon_budgets() -> AuthoringBudgets:
    """Load the budget the project's declaration points the engine at."""
    resource = PROJECT_DECLARATION.authoring_budget
    raw = (resource.root / "quillon.authoring_budget.json").read_bytes()
    document = load_authoring_budget_document(raw, field=resource.resource_id)
    return AuthoringBudgets.from_document(
        document,
        field=resource.resource_id,
        declared_layers=PROJECT_DECLARATION.labels("layer"),
    )


def quillon_kernel(budgets: AuthoringBudgets | None = None) -> EnvelopeKernel:
    """Return the engine bound to the fake project's declaration."""
    return EnvelopeKernel(
        declaration=PROJECT_DECLARATION,
        layout=QUILLON_LAYOUT,
        budgets=budgets if budgets is not None else load_quillon_budgets(),
        hooks=QUILLON_HOOKS,
        implementation=QUILLON_IMPLEMENTATION,
    )


# -- repository scaffolding used by the kernel tests ----------------------


def write_repo(root: Path) -> None:
    """Lay out a minimal quillon repository: two frozen bases, two envelopes."""
    studies = root / QUILLON_LAYOUT.envelope_directory
    bases = root / "bases"
    studies.mkdir(parents=True, exist_ok=True)
    bases.mkdir(parents=True, exist_ok=True)

    write_json(
        bases / "baseline.survey.json",
        {
            "schema_id": SURVEY_FAMILY,
            "name": "baseline",
            "settings": {"span": 4, "cadence": 2, "damping": 0.5},
        },
    )
    write_json(
        bases / "baseline.digest.json",
        {"schema_id": DIGEST_FAMILY, "name": "baseline-digest", "summary": "nothing yet"},
    )
    write_envelope(
        studies / f"widened{QUILLON_LAYOUT.envelope_suffix}",
        {
            "schema": "quillon.study.v1",
            "name": "widened",
            "layer": "survey",
            "base": "bases/baseline.survey.json",
            "assert": [{"path": "settings.cadence", "equals": 2}],
            "body": {"settings": {"span": 9}},
        },
    )
    write_envelope(
        studies / f"widened-digest{QUILLON_LAYOUT.envelope_suffix}",
        {
            "schema": "quillon.study.v1",
            "name": "widened-digest",
            "layer": "digest",
            "base": "bases/baseline.digest.json",
            "body": {"of": "widened", "summary": "the span was widened"},
        },
    )


def write_json(path: Path, document: Any) -> None:
    """Write one JSON document in the readable tracked form."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(document, indent=2) + "\n", encoding="utf-8")


def write_envelope(path: Path, document: Any) -> None:
    """Write one authored envelope in a form the pre-parse guards admit."""
    write_json(path, document)


__all__ = [
    "DIGEST_FAMILY",
    "QUILLON_HOOKS",
    "QUILLON_IMPLEMENTATION",
    "QUILLON_LAYOUT",
    "SURVEY_FAMILY",
    "authored_assertions",
    "layer_of",
    "load_quillon_budgets",
    "quillon_kernel",
    "write_envelope",
    "write_json",
    "write_repo",
]
