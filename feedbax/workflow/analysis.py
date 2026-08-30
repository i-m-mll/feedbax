"""Analysis authoring lowering into the finite workflow contract."""

from collections.abc import Mapping

from ._operation_lowering import operation
from .plan import Operation


def lower_analysis_operation(
    *, compiled_schema_id: str, semantic_hash: str, input_types: Mapping[str, str]
) -> Operation:
    return operation(
        type_id="feedbax.operation.analyze",
        compiled_schema_id=compiled_schema_id,
        semantic_hash=semantic_hash,
        input_types=input_types,
        determinism="deterministic",
        cache_policy="content_addressed",
        effect="pure",
    )
