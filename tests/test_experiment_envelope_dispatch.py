"""The one authoring entrypoint and its direct-dispatch contract.

``python -m feedbax preflight-experiment-envelope <envelope>`` is the single
documented authoring command. Feedbax compiles only its explicitly supported
authored dialects. There is no compiler seam. No registry mediates dispatch, no
plugin family carries compilers, no registration record binds a schema string to
a callable, and nothing can be injected between an authored envelope and the
document it compiles to.

These tests hold that line from both sides. A built-in dialect compiles with no
plugin argument at all, and an envelope declaring any other schema is refused by
name rather than being routed to something that might claim it.
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any

import pytest

import feedbax.plugins as plugins
from feedbax.__main__ import main
from feedbax.contracts.experiment_envelope import (
    ExperimentEnvelopeCompileResult,
    ExperimentEnvelopeCompilerError,
    ExperimentEnvelopeRejection,
    ExperimentEnvelopeRejectionCategory,
    dispatch_experiment_envelope,
    require_builtin_envelope_schema,
)
from feedbax.contracts.experiment_envelope_dialect import (
    EXPERIMENT_LAYER_ROOT_AUTHORITY_SCHEMA_ID,
    EXPERIMENT_LAYER_ROOT_AUTHORITY_SCHEMA_VERSION,
    EXPERIMENT_ENVELOPE_SCHEMA_VERSION,
    EXPERIMENT_ENVELOPE_SUPPORTED_SCHEMA_VERSIONS,
)
from feedbax.envelope import canonical_sha256
from feedbax.plugins.composition import compose_application

import tests.fake_project_experiment as fixture

FOREIGN_SCHEMA = "tests.fake_experiment.v1"


def _write(path: Path, payload: dict[str, Any]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return path


def _run(root: Path, alias: str, *extra: str) -> int:
    return main(
        [
            "preflight-experiment-envelope",
            str(fixture.envelope_path(root, alias)),
            "--repo-root",
            str(root),
            "--out-dir",
            fixture.OUTPUT_DIRECTORY,
            *extra,
        ]
    )


# --- the built-in dialect compiles with nothing registered -------------------


def test_the_entrypoint_compiles_the_builtin_dialect_with_no_plugins(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """No ``--plugin`` argument: the compiler is reached directly, not claimed."""
    fixture.write_repo(tmp_path)

    code = _run(tmp_path, "widened")

    assert code == 0
    payload = json.loads(capsys.readouterr().out)
    # The version the compiled document declared, not the version this build
    # calls current: quillon's training envelope is authored at v1 and compiles
    # as v1, so reporting the current constant would name the wrong grammar.
    assert payload["envelope_schema"] == fixture.TRAINING_ENVELOPE["schema"]
    assert payload["family"] == "training_run_matrix"
    generated = tmp_path / fixture.OUTPUT_DIRECTORY
    assert (generated / payload["compile_lock_path"]).is_file()
    assert (generated / payload["document_path"]).is_file()


def test_out_dir_override_is_honoured(tmp_path: Path) -> None:
    fixture.write_repo(tmp_path)

    code = main(
        [
            "preflight-experiment-envelope",
            str(fixture.envelope_path(tmp_path, "widened")),
            "--repo-root",
            str(tmp_path),
            "--out-dir",
            "elsewhere",
        ]
    )

    assert code == 0
    assert (tmp_path / "elsewhere" / "widened.compile-lock.json").is_file()


def test_rerunning_an_unchanged_envelope_rewrites_identical_bytes(
    tmp_path: Path,
) -> None:
    fixture.write_repo(tmp_path)
    assert _run(tmp_path, "widened") == 0
    lock = tmp_path / fixture.OUTPUT_DIRECTORY / "widened.compile-lock.json"
    first = lock.read_bytes()

    assert _run(tmp_path, "widened") == 0

    assert lock.read_bytes() == first


def test_entrypoint_dispatches_the_v4_comparison_policy_root(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    fixture.write_repo(tmp_path)
    authority = {
        "schema_id": EXPERIMENT_LAYER_ROOT_AUTHORITY_SCHEMA_ID,
        "schema_version": EXPERIMENT_LAYER_ROOT_AUTHORITY_SCHEMA_VERSION,
        "kind": "comparison_policy",
        "roles": {
            "reference": {
                "source_class": "quillon.loss_trace",
                "label": "Reference",
                "training_policy": "fixed",
            },
            "candidate": {
                "source_class": "quillon.loss_trace",
                "label": "Candidate",
                "training_policy": "adaptive",
                "figure_template": "terminal",
            },
        },
        "figure_templates": {
            "terminal": {
                "name": "terminal",
                "description": "Generic terminal comparison",
                "assembler": "quillon.comparison_grid",
            }
        },
        "comparison_policy": {
            "supported_source_class": "quillon.loss_trace",
            "required_cadence": "per_checkpoint",
            "required_equal_authority": ["training_data"],
            "mismatch_policy": "fail_closed",
        },
    }
    ref = "authorities/dispatch-comparison.json"
    fixture.write_json(tmp_path / ref, authority)
    fixture.write_envelope(
        fixture.envelope_path(tmp_path, "dispatch-comparison"),
        {
            "schema": EXPERIMENT_ENVELOPE_SCHEMA_VERSION,
            "name": "dispatch-comparison",
            "comparison": {"root": {"ref": ref, "sha256": canonical_sha256(authority)}},
        },
    )

    assert _run(tmp_path, "dispatch-comparison") == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["family"] == "comparison_policy"
    assert payload["envelope_schema"] == EXPERIMENT_ENVELOPE_SCHEMA_VERSION
    document = json.loads(
        (tmp_path / fixture.OUTPUT_DIRECTORY / payload["document_path"]).read_text()
    )
    assert document["schema_id"] == "feedbax.spec.comparison_policy"
    assert document["schema_version"] == "feedbax.spec.comparison_policy.v1"
    assert document["name"] == "dispatch-comparison"


# --- an unsupported schema is refused by name -------------------------------


def test_a_foreign_schema_is_rejected_naming_both_schemas(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """No compiler can claim another schema, so the refusal is the whole answer."""
    fixture.write_repo(tmp_path)
    _write(
        fixture.envelope_path(tmp_path, "widened"),
        {**fixture.TRAINING_ENVELOPE, "schema": FOREIGN_SCHEMA},
    )

    code = _run(tmp_path, "widened")

    assert code == 2
    stderr = capsys.readouterr().err
    assert "category=unsupported-schema-version" in stderr
    assert "field=schema" in stderr
    assert FOREIGN_SCHEMA in stderr
    assert EXPERIMENT_ENVELOPE_SCHEMA_VERSION in stderr
    assert not (tmp_path / fixture.OUTPUT_DIRECTORY).exists()


def test_require_builtin_envelope_schema_accepts_only_supported_dialects() -> None:
    for supported in EXPERIMENT_ENVELOPE_SUPPORTED_SCHEMA_VERSIONS:
        require_builtin_envelope_schema(supported)

    for schema in (FOREIGN_SCHEMA, ""):
        with pytest.raises(ExperimentEnvelopeRejection) as caught:
            require_builtin_envelope_schema(schema)
        assert caught.value.category is (
            ExperimentEnvelopeRejectionCategory.UNSUPPORTED_SCHEMA_VERSION
        )
        assert caught.value.field == "schema"


def test_envelope_without_a_schema_is_rejected(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    fixture.write_repo(tmp_path)
    stated = dict(fixture.TRAINING_ENVELOPE)
    stated.pop("schema")
    _write(fixture.envelope_path(tmp_path, "widened"), stated)

    code = _run(tmp_path, "widened")

    assert code == 2
    assert "field=schema" in capsys.readouterr().err


def test_missing_envelope_file_is_infrastructure_failure(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    code = main(
        [
            "preflight-experiment-envelope",
            str(tmp_path / "absent.json"),
            "--repo-root",
            str(tmp_path),
        ]
    )

    assert code == 1
    assert "cannot read experiment envelope" in capsys.readouterr().err


# --- the seam itself is gone -------------------------------------------------


@pytest.mark.parametrize(
    "name",
    [
        "ExperimentEnvelopeCompiler",
        "ExperimentEnvelopeCompilerRegistration",
        "ExperimentEnvelopeCompilerRegistry",
        "ExperimentEnvelopeCompilerCollisionError",
    ],
)
def test_no_compiler_registry_symbol_survives(name: str) -> None:
    """A registrable compiler is a second dialect waiting to happen."""
    import feedbax.contracts.experiment_envelope as contract
    import feedbax.envelope as envelope

    for module in (contract, plugins, envelope):
        assert not hasattr(module, name), f"{module.__name__} still exposes {name}"
        assert name not in getattr(module, "__all__", ())


def test_the_bootstrap_has_no_envelope_compiler_family() -> None:
    state = asyncio.run(compose_application(modules=(), local_component_source=None))

    assert not hasattr(state.bundle, "experiment_envelope_compilers")
    families = {key.family for key in plugins.APPLICATION_REGISTRY_KEYS}
    assert "experiment_envelope_compilers" not in families
    assert "EXPERIMENT_ENVELOPE_COMPILERS" not in plugins.__all__


# --- the dispatcher still guards what it hands back --------------------------


def test_dispatch_refuses_a_compiler_result_that_claims_another_schema(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The one compiler is trusted to compile, not to rename the dialect."""
    import feedbax.envelope.entrypoint as entrypoint

    monkeypatch.setattr(
        entrypoint,
        "compile_experiment_envelope",
        lambda request: ExperimentEnvelopeCompileResult(
            envelope_schema=FOREIGN_SCHEMA,
            name="x",
            family="f",
            compile_lock_path="a.json",
            document_path="b.json",
        ),
    )

    with pytest.raises(ExperimentEnvelopeCompilerError, match="reported envelope schema"):
        dispatch_experiment_envelope(
            {"schema": EXPERIMENT_ENVELOPE_SCHEMA_VERSION},
            envelope_path=tmp_path / "e.json",
            repo_root=tmp_path,
            out_dir=tmp_path,
        )


def test_undeclared_outputs_are_infrastructure_failure(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    import feedbax.envelope.entrypoint as entrypoint

    def _lying_compile(request):
        request.out_dir.mkdir(parents=True, exist_ok=True)
        return ExperimentEnvelopeCompileResult(
            envelope_schema=fixture.TRAINING_ENVELOPE["schema"],
            name="widened",
            family="training_run_matrix",
            compile_lock_path="never-written.json",
            document_path="also-never-written.json",
        )

    fixture.write_repo(tmp_path)
    monkeypatch.setattr(entrypoint, "compile_experiment_envelope", _lying_compile)

    code = _run(tmp_path, "widened")

    assert code == 1
    assert "outputs it did not write" in capsys.readouterr().err


def test_compiling_the_dialect_without_a_project_declaration_is_infrastructure() -> None:
    from feedbax.contracts.experiment_envelope import ExperimentEnvelopeCompileRequest
    from feedbax.envelope.entrypoint import compile_experiment_envelope

    with pytest.raises(ExperimentEnvelopeCompilerError, match="declaration of the project"):
        compile_experiment_envelope(
            ExperimentEnvelopeCompileRequest(
                envelope={},
                envelope_path=Path("studies/x.envelope.json"),
                repo_root=Path("."),
                out_dir=Path("."),
            )
        )


# --- the rejection vocabulary ------------------------------------------------


def test_rejection_categories_are_a_closed_set() -> None:
    """The vocabulary is one closed set shared by the dispatcher and the kernel.

    Categories are added deliberately, in the change that gives the engine a
    reason to name one. What the set forbids is a failure inventing a category
    string of its own at the point of failure.
    """
    assert {item.value for item in ExperimentEnvelopeRejectionCategory} == {
        "unknown-field",
        "missing-field",
        "invalid-value",
        "duplicate-key",
        "noncanonical-format",
        "echoed-inherited-value",
        "derived-value-authored",
        "budget-exceeded",
        "assertion-failed",
        "illegal-assertion-path",
        "unresolved-row-key",
        "empty-selection",
        "unsupported-schema-version",
        "unresolved-base",
        "cross-family-base",
        "retired-base-family",
        "unresolved-upstream-reference",
        "co-created-protected-document",
    }
    with pytest.raises(ValueError):
        ExperimentEnvelopeRejection("invented-category", "no")


def test_compile_result_rejects_unknown_schema_versions() -> None:
    payload = ExperimentEnvelopeCompileResult(
        envelope_schema=EXPERIMENT_ENVELOPE_SCHEMA_VERSION,
        name="x",
        family="f",
        compile_lock_path="a.json",
        document_path="b.json",
    ).model_dump(mode="json")

    for bad in (
        "feedbax.spec.experiment_envelope_compile_result.v0",
        "feedbax.spec.experiment_envelope_compile_result.v2",
    ):
        with pytest.raises(ValueError, match="unsupported"):
            ExperimentEnvelopeCompileResult.model_validate({**payload, "schema_version": bad})
