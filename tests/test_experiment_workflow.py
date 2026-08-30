"""The public workflow entry points: one API call, one CLI subcommand.

``execute_experiment_workflow`` takes which finite workflow to execute and where to fulfill it,
and nothing else — no lowering, no payload preparer, no omission applier. The
CLI is the same operation from a shell, with the documented exit codes: ``0``
executed, ``2`` a stable typed rejection, ``1`` infrastructure.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from feedbax.analysis.evaluation import EvaluationRecipeResult
from feedbax.analysis.fulfillment_adapters import FulfillmentEnvironment
from feedbax.workflow.derivation import CompiledOutputError
from feedbax.workflow.execution import ExternalOperationError
from feedbax.workflow.experiment import (
    execute_experiment_workflow,
    plan_experiment_workflow,
)
from feedbax.analysis.reports import REPORT_RENDER_ROLE, ReportRecipeResult
from feedbax.contracts.experiment_compile_lock import EvaluationSubjectBinding, ReportParentBinding
from feedbax.contracts.manifest import load_manifest, store_bytes_artifact

from tests.fake_project_experiment.products import (
    BULLETIN_TYPE,
    PROBE_TYPE,
    QuillonOutputs,
    planned,
)


@pytest.fixture
def outputs(tmp_path: Path) -> QuillonOutputs:
    return QuillonOutputs(tmp_path / "repo")


@pytest.fixture
def environment(tmp_path: Path, application_registry_bundle):
    def evaluation_recipe(run_spec, root, states_path, execution_context):
        artifact = store_bytes_artifact(
            f"{run_spec.params.get('stage', '')}\n".encode(),
            root=root,
            role="evaluation_states",
            logical_name="states.bin",
        )
        return EvaluationRecipeResult(
            states=None,
            summary_metrics={"stage": run_spec.params.get("stage", "")},
            artifacts=[artifact],
            metadata={"states_schema": "quillon.states.v1"},
        )

    def report_recipe(report_spec, root, inputs):
        artifact = store_bytes_artifact(
            f"# {report_spec.report_type}\n".encode(),
            root=root,
            role=REPORT_RENDER_ROLE,
            logical_name="bulletin.md",
            media_type="text/markdown",
            suffix=".md",
        )
        return ReportRecipeResult(artifacts=[artifact], summary={"inputs": len(inputs)})

    application_registry_bundle.evaluation_recipes.register(PROBE_TYPE, evaluation_recipe)
    application_registry_bundle.report_recipes.register(BULLETIN_TYPE, report_recipe)
    return FulfillmentEnvironment(
        root=tmp_path / "receipts", registries=application_registry_bundle
    )


def _pair(outputs: QuillonOutputs) -> str:
    probe = outputs.probe("study")
    outputs.bulletin(
        "study-bulletin",
        references=[
            planned(
                probe,
                role_path="body.of",
                consumer=ReportParentBinding(parent_kind="probe", parent_id="study"),
            )
        ],
    )
    return "study-bulletin"


def test_one_call_fulfils_the_whole_closure(
    outputs: QuillonOutputs, environment: FulfillmentEnvironment
) -> None:
    target = _pair(outputs)
    result = execute_experiment_workflow(
        target, output_directory=outputs.output_directory, environment=environment
    )
    assert result.summary()["order"] == ["evaluation:study", "report:study-bulletin"]
    assert result.summary()["executed"] == ["evaluation:study", "report:study-bulletin"]
    assert result.summary()["reused"] == []
    for entry in result.run.results:
        assert load_manifest(entry.receipt.path).status == "completed"

    again = execute_experiment_workflow(
        target, output_directory=outputs.output_directory, environment=environment
    )
    assert again.summary()["executed"] == []
    assert again.summary()["reused"] == result.summary()["order"]


def test_planning_reads_nothing_it_does_not_need_and_writes_nothing(
    outputs: QuillonOutputs, tmp_path: Path
) -> None:
    target = _pair(outputs)
    plan, index = plan_experiment_workflow(target, output_directory=outputs.output_directory)
    assert plan.target.text == "report:study-bulletin"
    assert len(index.envelopes) == 2
    assert not (tmp_path / "receipts").exists()


def test_a_closure_still_needing_training_refuses(
    outputs: QuillonOutputs, environment: FulfillmentEnvironment
) -> None:
    cohort = outputs.cohort("sweep")
    outputs.probe(
        "sweep-probe",
        references=[
            planned(
                cohort,
                role_path="body.subject",
                consumer=EvaluationSubjectBinding(subject_id="sweep"),
            )
        ],
    )
    with pytest.raises(ExternalOperationError):
        execute_experiment_workflow(
            "sweep-probe",
            output_directory=outputs.output_directory,
            environment=environment,
        )
    assert not environment.root.exists()


def test_an_output_directory_that_is_not_one_refuses(
    tmp_path: Path, environment: FulfillmentEnvironment
) -> None:
    with pytest.raises(CompiledOutputError, match="not a directory"):
        execute_experiment_workflow(
            "anything", output_directory=tmp_path / "absent", environment=environment
        )


# --------------------------------------------------------------------------
# The CLI, and its documented exit codes
# --------------------------------------------------------------------------


def _cli(outputs: QuillonOutputs, target: str, receipt_root: Path, plugin: str) -> int:
    from feedbax.__main__ import main

    return main(
        [
            "execute-experiment-workflow",
            target,
            "--out-dir",
            str(outputs.output_directory),
            "--repo-root",
            str(outputs.root),
            "--receipt-root",
            str(receipt_root),
            "--plugin",
            plugin,
        ]
    )


def test_the_cli_exits_zero_and_prints_the_walk(
    outputs: QuillonOutputs, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    target = _pair(outputs)
    code = _cli(outputs, target, tmp_path / "cli-receipts", "tests.workflow_cli_plugin")
    assert code == 0
    summary = json.loads(capsys.readouterr().out)
    assert summary["target"] == "report:study-bulletin"
    assert summary["executed"] == ["evaluation:study", "report:study-bulletin"]


def test_the_cli_exits_two_on_a_stable_typed_rejection(
    outputs: QuillonOutputs, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    outputs.probe("known")
    code = _cli(outputs, "unknown", tmp_path / "cli-receipts", "tests.workflow_cli_plugin")
    assert code == 2
    assert "CompiledOutputError" in capsys.readouterr().err


def test_the_cli_exits_two_when_the_closure_still_needs_training(
    outputs: QuillonOutputs, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    cohort = outputs.cohort("sweep")
    outputs.probe(
        "sweep-probe",
        references=[
            planned(
                cohort,
                role_path="body.subject",
                consumer=EvaluationSubjectBinding(subject_id="sweep"),
            )
        ],
    )
    code = _cli(outputs, "sweep-probe", tmp_path / "cli-receipts", "tests.workflow_cli_plugin")
    assert code == 2
    assert "ExternalOperationError" in capsys.readouterr().err


# --------------------------------------------------------------------------
# The staged-input surface: one descriptor, and roots bound by name
# --------------------------------------------------------------------------


def _descriptor_document(*provider_names: str) -> dict:
    from feedbax.contracts.artifact_custody import ImmutableArtifactBlobProviderSpec
    from feedbax.contracts.staged_execution import (
        STAGED_EXECUTION_DESCRIPTOR_SCHEMA_ID,
        STAGED_EXECUTION_DESCRIPTOR_SCHEMA_VERSION,
    )

    provider = ImmutableArtifactBlobProviderSpec().model_dump(mode="json")
    return {
        "schema_id": STAGED_EXECUTION_DESCRIPTOR_SCHEMA_ID,
        "schema_version": STAGED_EXECUTION_DESCRIPTOR_SCHEMA_VERSION,
        "artifact_providers": {name: provider for name in provider_names},
        "checkpoint_custody": {},
    }


def _write_descriptor(tmp_path: Path, *provider_names: str) -> Path:
    path = tmp_path / "execution-descriptor.json"
    path.write_text(json.dumps(_descriptor_document(*provider_names)), encoding="utf-8")
    return path


def _fulfill_cli(outputs: QuillonOutputs, target: str, receipt_root: Path, *extra: str) -> int:
    from feedbax.__main__ import main

    return main(
        [
            "execute-experiment-workflow",
            target,
            "--out-dir",
            str(outputs.output_directory),
            "--repo-root",
            str(outputs.root),
            "--receipt-root",
            str(receipt_root),
            "--plugin",
            "tests.workflow_cli_plugin",
            *extra,
        ]
    )


def test_the_staged_flags_parse_repeat_and_bind_by_name(
    outputs: QuillonOutputs, tmp_path: Path
) -> None:
    """Four flags, three of them repeatable, all of them ``NAME=ROOT``.

    The names are the descriptor's own, and the run is refused before anything
    executes when a bound name is not one it declares — which is the check that
    makes ``NAME=ROOT`` mean something rather than being free-form text.
    """
    target = _pair(outputs)
    descriptor = _write_descriptor(tmp_path, "results", "evidence.backup")
    providers = {
        "results": tmp_path / "provider-results",
        "evidence.backup": tmp_path / "provider-backup",
    }
    retained = {"primary": tmp_path / "retained-a", "secondary": tmp_path / "retained-b"}
    for root in (*providers.values(), *retained.values()):
        root.mkdir()

    code = _fulfill_cli(
        outputs,
        target,
        tmp_path / "receipts",
        "--execution-descriptor",
        str(descriptor),
        *[f"--artifact-provider={name}={root}" for name, root in providers.items()],
        *[f"--manifest-root={name}={root}" for name, root in retained.items()],
    )
    assert code == 0


def test_a_bound_root_the_descriptor_never_declares_refuses(
    outputs: QuillonOutputs, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    target = _pair(outputs)
    descriptor = _write_descriptor(tmp_path, "results")
    stray = tmp_path / "stray"
    stray.mkdir()

    code = _fulfill_cli(
        outputs,
        target,
        tmp_path / "receipts",
        "--execution-descriptor",
        str(descriptor),
        f"--artifact-provider=results={stray}",
        f"--artifact-provider=unknown={stray}",
    )
    assert code == 2
    assert "must exactly match the descriptor" in capsys.readouterr().err
    assert not (tmp_path / "receipts").exists()


@pytest.mark.parametrize("flag", ["--artifact-provider", "--manifest-root", "--checkpoint-custody"])
def test_a_binding_flag_without_a_descriptor_refuses_before_fulfillment(
    outputs: QuillonOutputs, tmp_path: Path, capsys: pytest.CaptureFixture[str], flag: str
) -> None:
    """A root bound to a name no descriptor declares is a root nobody asked for."""
    target = _pair(outputs)
    code = _fulfill_cli(outputs, target, tmp_path / "receipts", f"{flag}=named={tmp_path}")
    assert code == 2
    assert "require --execution-descriptor" in capsys.readouterr().err
    assert not (tmp_path / "receipts").exists()


@pytest.mark.parametrize("value", ["noequals", "=root", "name="])
def test_a_binding_that_is_not_name_equals_root_refuses(
    outputs: QuillonOutputs, tmp_path: Path, capsys: pytest.CaptureFixture[str], value: str
) -> None:
    target = _pair(outputs)
    code = _fulfill_cli(
        outputs,
        target,
        tmp_path / "receipts",
        "--execution-descriptor",
        str(_write_descriptor(tmp_path)),
        f"--manifest-root={value}",
    )
    assert code == 2
    assert "NAME=ROOT" in capsys.readouterr().err
    assert not (tmp_path / "receipts").exists()


def test_a_descriptor_alone_is_a_complete_declaration(
    outputs: QuillonOutputs, tmp_path: Path
) -> None:
    """A descriptor declaring no authority binds no root, and that is valid."""
    target = _pair(outputs)
    code = _fulfill_cli(
        outputs,
        target,
        tmp_path / "receipts",
        "--execution-descriptor",
        str(_write_descriptor(tmp_path)),
    )
    assert code == 0


def test_the_receipt_root_is_never_read_as_a_provider_root(
    outputs: QuillonOutputs, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """A declared provider stays unbound unless a root is bound to its name.

    The receipt root is output and admission custody. Letting it stand in for a
    provider the descriptor declares would merge two custody domains the caller
    deliberately kept apart, so the missing binding refuses instead.
    """
    target = _pair(outputs)
    code = _fulfill_cli(
        outputs,
        target,
        tmp_path / "receipts",
        "--execution-descriptor",
        str(_write_descriptor(tmp_path, "results")),
    )
    assert code == 2
    error = capsys.readouterr().err
    assert "must exactly match the descriptor" in error
    assert "missing=['results']" in error
