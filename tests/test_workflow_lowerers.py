"""Architecture evidence for the five layer-owned workflow lowerers."""

from pathlib import Path

from feedbax.contracts.experiment_compile_lock import (
    AnalysisInputBinding,
    EvaluationSubjectBinding,
    FigureRuntimeInputBinding,
    ReportParentBinding,
)
from feedbax.workflow.analysis import lower_analysis_operation
from feedbax.workflow.campaign import lower_campaign_operation
from feedbax.workflow.derivation import derive_workflow_plan, read_compiled_outputs
from feedbax.workflow.evaluation import lower_evaluation_operation
from feedbax.workflow.fulfillment import lower_fulfillment_operation
from feedbax.workflow.plan import workflow_plan_from_document
from feedbax.workflow.report import lower_report_operation

from tests.fake_project_experiment.products import QuillonOutputs, planned


def test_each_authoring_domain_owns_a_distinct_lowerer() -> None:
    lowerers = (
        lower_campaign_operation,
        lower_evaluation_operation,
        lower_analysis_operation,
        lower_report_operation,
        lower_fulfillment_operation,
    )
    operations = tuple(
        lowerer(
            compiled_schema_id=f"example.schema.{index}", semantic_hash="a" * 64, input_types={}
        )
        for index, lowerer in enumerate(lowerers)
    )
    assert [operation.type_id for operation in operations] == [
        "feedbax.operation.train",
        "feedbax.operation.evaluate",
        "feedbax.operation.analyze",
        "feedbax.operation.report",
        "feedbax.operation.render",
    ]
    assert all(
        set(operation.parameters) == {"compiled_schema_id", "semantic_hash"}
        for operation in operations
    )


def test_sisu_exemplar_lowers_end_to_end_into_one_workflow_plan(tmp_path: Path) -> None:
    outputs = QuillonOutputs(tmp_path / "sisu-exemplar")
    campaign = outputs.cohort("sisu-continuous-conditioning")
    evaluation = outputs.probe(
        "sisu-evaluation",
        references=[
            planned(
                campaign,
                role_path="subject.model",
                consumer=EvaluationSubjectBinding(subject_id="conditioned-controller"),
            )
        ],
    )
    analysis = outputs.condensate(
        "sisu-analysis",
        references=[
            planned(
                evaluation,
                role_path="inputs.evaluation",
                consumer=AnalysisInputBinding(alias="evaluation", role="observed"),
            )
        ],
    )
    figure = outputs.plate(
        "sisu-figure",
        references=[
            planned(
                analysis,
                role_path="runtime.analysis",
                consumer=FigureRuntimeInputBinding(input_role="analysis"),
            )
        ],
    )
    outputs.bulletin(
        "sisu-report",
        references=[
            planned(
                figure,
                role_path="body.figure",
                consumer=ReportParentBinding(parent_kind="figure", parent_id="sisu-figure"),
            )
        ],
    )

    plan = derive_workflow_plan(
        read_compiled_outputs(outputs.output_directory), target="sisu-report"
    )
    assert [node.key.layer for node in plan.nodes] == [
        "campaign",
        "evaluation",
        "analysis",
        "figure",
        "report",
    ]
    assert [node.operation.type_id for node in plan.nodes] == [
        "feedbax.operation.train",
        "feedbax.operation.evaluate",
        "feedbax.operation.analyze",
        "feedbax.operation.render",
        "feedbax.operation.report",
    ]
    assert workflow_plan_from_document(plan.document()).identity == plan.identity
