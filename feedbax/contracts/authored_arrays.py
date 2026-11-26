"""Scalar budget for values written directly into authored project files."""

from pydantic import JsonValue

from feedbax.contracts.project_sources import pointer

AUTHORED_ARRAY_SCALAR_LIMIT = 256


class AuthoredArrayBudgetError(ValueError):
    """One authored array value exceeds the scalar budget."""


def check_authored_array_budget(
    value: JsonValue, *, path: tuple[str, ...] = (), source: str | None = None,
) -> None:
    """Count nested arrays as one value. A mapping starts a separate value at each key."""

    def visit(node: JsonValue, at: tuple[str, ...], *, within_array: bool = False) -> int:
        if isinstance(node, dict):
            for key, child in node.items():
                visit(child, (*at, key))
            return 0
        if isinstance(node, list):
            count = sum(visit(child, (*at, str(index)), within_array=True) for index, child in enumerate(node))
            if not within_array and count > AUTHORED_ARRAY_SCALAR_LIMIT:
                location = pointer(at) or "<root>"
                if source is not None:
                    location = f"{source} at {location}"
                raise AuthoredArrayBudgetError(
                    f"Array at {location} has {count} scalars. The authored limit is "
                    f"{AUTHORED_ARRAY_SCALAR_LIMIT}. Use a rule that produces the array or a content reference."
                )
            return count
        return 1

    visit(value, path)
