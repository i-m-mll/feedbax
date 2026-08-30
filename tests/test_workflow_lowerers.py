"""Architecture evidence for the five layer-owned workflow lowerers."""

from feedbax.workflow.analysis import lower_analysis_operation
from feedbax.workflow.campaign import lower_campaign_operation
from feedbax.workflow.evaluation import lower_evaluation_operation
from feedbax.workflow.fulfillment import lower_fulfillment_operation
from feedbax.workflow.report import lower_report_operation


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
