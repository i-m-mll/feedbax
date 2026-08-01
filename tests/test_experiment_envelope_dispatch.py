"""The one generic authoring entrypoint and its downstream dispatch contract.

``python -m feedbax preflight-experiment-envelope <envelope>`` is the single
documented authoring command. Feedbax knows no downstream dialect: it reads the
envelope's declared schema, finds the compiler that claimed that schema through
the ordinary plugin bootstrap with injected registries, and turns the outcome
into the documented exit codes.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from feedbax.__main__ import main
from feedbax.contracts.experiment_envelope import (
    ExperimentEnvelopeCompileResult,
    ExperimentEnvelopeCompilerCollisionError,
    ExperimentEnvelopeCompilerError,
    ExperimentEnvelopeCompilerRegistration,
    ExperimentEnvelopeCompilerRegistry,
    ExperimentEnvelopeRejection,
    ExperimentEnvelopeRejectionCategory,
    dispatch_experiment_envelope,
)

PLUGIN_MODULE = "tests.experiment_envelope_fake_compiler"
ENVELOPE_SCHEMA = "tests.fake_experiment.v1"


def _write(path: Path, payload: dict[str, Any]) -> Path:
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return path


def _run(tmp_path: Path, envelope: dict[str, Any], *extra: str) -> int:
    envelope_path = _write(tmp_path / "envelope.json", envelope)
    return main(
        [
            "preflight-experiment-envelope",
            str(envelope_path),
            "--repo-root",
            str(tmp_path),
            "--plugin",
            PLUGIN_MODULE,
            *extra,
        ]
    )


def test_entrypoint_dispatches_on_the_declared_envelope_schema(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    code = _run(tmp_path, {"schema": ENVELOPE_SCHEMA, "name": "fake-run"})

    assert code == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["envelope_schema"] == ENVELOPE_SCHEMA
    assert payload["family"] == "fake_document"
    assert (tmp_path / "generated" / "fake-run.compile-lock.json").is_file()
    assert (tmp_path / "generated" / "fake-run.fake_document.json").is_file()


def test_out_dir_override_is_honoured(tmp_path: Path) -> None:
    code = _run(
        tmp_path,
        {"schema": ENVELOPE_SCHEMA, "name": "fake-run"},
        "--out-dir",
        "elsewhere",
    )

    assert code == 0
    assert (tmp_path / "elsewhere" / "fake-run.compile-lock.json").is_file()


def test_rerunning_an_unchanged_envelope_rewrites_identical_bytes(tmp_path: Path) -> None:
    envelope = {"schema": ENVELOPE_SCHEMA, "name": "fake-run"}
    assert _run(tmp_path, envelope) == 0
    first = (tmp_path / "generated" / "fake-run.compile-lock.json").read_bytes()
    assert _run(tmp_path, envelope) == 0

    assert (tmp_path / "generated" / "fake-run.compile-lock.json").read_bytes() == first


def test_rejection_exits_two_and_names_field_category_and_home(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    code = _run(
        tmp_path,
        {"schema": ENVELOPE_SCHEMA, "name": "fake-run", "metadata": {"note": "prose"}},
    )

    assert code == 2
    stderr = capsys.readouterr().err
    assert "category=unknown-field" in stderr
    assert "field=metadata" in stderr
    assert "correct home:" in stderr


def test_envelope_without_a_schema_is_rejected(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    code = _run(tmp_path, {"name": "fake-run"})

    assert code == 2
    assert "field=schema" in capsys.readouterr().err


def test_unclaimed_schema_is_infrastructure_failure(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    code = _run(tmp_path, {"schema": "tests.unclaimed.v1", "name": "fake-run"})

    assert code == 1
    assert "no registered compiler claims" in capsys.readouterr().err


def test_missing_envelope_file_is_infrastructure_failure(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    code = main(
        [
            "preflight-experiment-envelope",
            str(tmp_path / "absent.json"),
            "--repo-root",
            str(tmp_path),
            "--plugin",
            PLUGIN_MODULE,
        ]
    )

    assert code == 1
    assert "cannot read experiment envelope" in capsys.readouterr().err


def test_compiler_is_only_reachable_through_the_injected_registry(tmp_path: Path) -> None:
    code = main(
        [
            "preflight-experiment-envelope",
            str(_write(tmp_path / "e.json", {"schema": ENVELOPE_SCHEMA, "name": "x"})),
            "--repo-root",
            str(tmp_path),
        ]
    )

    assert code == 1


def test_registry_refuses_two_claims_on_one_schema() -> None:
    registry = ExperimentEnvelopeCompilerRegistry()
    registration = ExperimentEnvelopeCompilerRegistration(
        envelope_schema=ENVELOPE_SCHEMA,
        owner="tests.first",
        compile=lambda request: None,
    )
    registry.register(registration)

    with pytest.raises(ExperimentEnvelopeCompilerCollisionError):
        registry.register(
            ExperimentEnvelopeCompilerRegistration(
                envelope_schema=ENVELOPE_SCHEMA,
                owner="tests.second",
                compile=lambda request: None,
            )
        )
    assert registry.available_keys() == (ENVELOPE_SCHEMA,)


def test_sealed_registry_refuses_late_registration() -> None:
    registry = ExperimentEnvelopeCompilerRegistry()
    registry.seal()

    with pytest.raises(RuntimeError, match="sealed"):
        registry.register(
            ExperimentEnvelopeCompilerRegistration(
                envelope_schema=ENVELOPE_SCHEMA,
                owner="tests.late",
                compile=lambda request: None,
            )
        )


def test_dispatch_refuses_a_result_that_claims_another_schema(tmp_path: Path) -> None:
    registry = ExperimentEnvelopeCompilerRegistry()
    registry.register(
        ExperimentEnvelopeCompilerRegistration(
            envelope_schema=ENVELOPE_SCHEMA,
            owner="tests.liar",
            compile=lambda request: ExperimentEnvelopeCompileResult(
                envelope_schema="tests.other.v1",
                name="x",
                family="f",
                compile_lock_path="a.json",
                document_path="b.json",
            ),
        )
    )

    with pytest.raises(ExperimentEnvelopeCompilerError, match="reported envelope schema"):
        dispatch_experiment_envelope(
            {"schema": ENVELOPE_SCHEMA},
            registry,
            envelope_path=tmp_path / "e.json",
            repo_root=tmp_path,
            out_dir=tmp_path,
        )


def test_undeclared_outputs_are_infrastructure_failure(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    import tests.experiment_envelope_fake_compiler as fake

    def _lying_compile(request):
        request.out_dir.mkdir(parents=True, exist_ok=True)
        return ExperimentEnvelopeCompileResult(
            envelope_schema=ENVELOPE_SCHEMA,
            name="fake-run",
            family="fake_document",
            compile_lock_path="never-written.json",
            document_path="also-never-written.json",
        )

    monkeypatch.setattr(fake, "compile_fake_envelope", _lying_compile)
    monkeypatch.setattr(
        fake,
        "PLUGIN_REGISTRATION",
        fake.PluginRegistration(
            fake.PluginDeclaration(
                "tests.experiment_envelope_fake_compiler",
                "1.0",
                1,
                families=(fake.FamilyRequirement("experiment_envelope_compilers"),),
            ),
            lambda context: context.registry(
                fake.EXPERIMENT_ENVELOPE_COMPILERS
            ).register(
                ExperimentEnvelopeCompilerRegistration(
                    envelope_schema=ENVELOPE_SCHEMA,
                    owner="tests.lying",
                    compile=_lying_compile,
                )
            ),
        ),
    )

    code = _run(tmp_path, {"schema": ENVELOPE_SCHEMA, "name": "fake-run"})

    assert code == 1
    assert "outputs it did not write" in capsys.readouterr().err


def test_rejection_categories_are_a_closed_set() -> None:
    """The vocabulary is one closed set shared by the dispatcher and the kernel.

    Categories are added deliberately, in the change that gives the engine a
    reason to name one. What the set forbids is a compiler inventing a category
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
        "unresolved-extension-label",
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
        envelope_schema=ENVELOPE_SCHEMA,
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
