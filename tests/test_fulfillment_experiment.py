"""The public fulfillment entry points: one API call, one CLI subcommand.

``fulfill_experiment_envelope`` takes what to fulfill and where to fulfill it,
and nothing else — no lowering, no payload preparer, no omission applier. The
CLI is the same operation from a shell, with the documented exit codes: ``0``
fulfilled, ``2`` a stable typed rejection, ``1`` infrastructure.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from feedbax.analysis.evaluation import EvaluationRecipeResult
from feedbax.analysis.fulfillment_adapters import FulfillmentEnvironment
from feedbax.analysis.fulfillment_derivation import CompiledOutputError
from feedbax.analysis.fulfillment_driver import ExternalBoundaryError
from feedbax.analysis.fulfillment_experiment import (
    fulfill_experiment_envelope,
    plan_experiment_envelope,
)
from feedbax.analysis.reports import REPORT_RENDER_ROLE, ReportRecipeResult
from feedbax.contracts.experiment_compile_lock import EvaluationSubjectBinding, ReportParentBinding
from feedbax.contracts.manifest import load_manifest, store_bytes_artifact

from tests.fake_project_extension.products import (
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
    result = fulfill_experiment_envelope(
        target, output_directory=outputs.output_directory, environment=environment
    )
    assert result.summary()["order"] == ["evaluation:study", "report:study-bulletin"]
    assert result.summary()["executed"] == ["evaluation:study", "report:study-bulletin"]
    assert result.summary()["reused"] == []
    for entry in result.run.results:
        assert load_manifest(entry.receipt.path).status == "completed"

    again = fulfill_experiment_envelope(
        target, output_directory=outputs.output_directory, environment=environment
    )
    assert again.summary()["executed"] == []
    assert again.summary()["reused"] == result.summary()["order"]


def test_planning_reads_nothing_it_does_not_need_and_writes_nothing(
    outputs: QuillonOutputs, tmp_path: Path
) -> None:
    target = _pair(outputs)
    plan, index = plan_experiment_envelope(target, output_directory=outputs.output_directory)
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
    with pytest.raises(ExternalBoundaryError):
        fulfill_experiment_envelope(
            "sweep-probe",
            output_directory=outputs.output_directory,
            environment=environment,
        )
    assert not environment.root.exists()


def test_an_output_directory_that_is_not_one_refuses(
    tmp_path: Path, environment: FulfillmentEnvironment
) -> None:
    with pytest.raises(CompiledOutputError, match="not a directory"):
        fulfill_experiment_envelope(
            "anything", output_directory=tmp_path / "absent", environment=environment
        )


# --------------------------------------------------------------------------
# The CLI, and its documented exit codes
# --------------------------------------------------------------------------


def _cli(outputs: QuillonOutputs, target: str, receipt_root: Path, plugin: str) -> int:
    from feedbax.__main__ import main

    return main(
        [
            "fulfill-experiment-envelope",
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
    code = _cli(outputs, target, tmp_path / "cli-receipts", "tests.fulfillment_cli_plugin")
    assert code == 0
    summary = json.loads(capsys.readouterr().out)
    assert summary["target"] == "report:study-bulletin"
    assert summary["executed"] == ["evaluation:study", "report:study-bulletin"]


def test_the_cli_exits_two_on_a_stable_typed_rejection(
    outputs: QuillonOutputs, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    outputs.probe("known")
    code = _cli(outputs, "unknown", tmp_path / "cli-receipts", "tests.fulfillment_cli_plugin")
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
    code = _cli(
        outputs, "sweep-probe", tmp_path / "cli-receipts", "tests.fulfillment_cli_plugin"
    )
    assert code == 2
    assert "ExternalBoundaryError" in capsys.readouterr().err
