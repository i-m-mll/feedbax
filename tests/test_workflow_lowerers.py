"""Architecture evidence for the declarative workflow operation authority."""

from feedbax.workflow.derivation import OPERATION_METADATA, lower_operation


def test_operation_metadata_is_the_only_layer_specific_lowering_authority() -> None:
    layers = ("campaign", "evaluation", "analysis", "report", "figure")
    operations = tuple(
        lower_operation(
            layer,
            compiled_schema_id=f"example.schema.{index}",
            semantic_hash="a" * 64,
            input_types={},
        )
        for index, layer in enumerate(layers)
    )
    assert tuple(OPERATION_METADATA) == ("campaign", "evaluation", "analysis", "figure", "report")
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
