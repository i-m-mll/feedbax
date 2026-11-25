"""The typed point operands a measure relates or uses alone."""

from dataclasses import dataclass

from feedbax.contracts.expression_check import Checked
from feedbax.contracts.expression_values import CallValue, Expression, NodePath, children


@dataclass(frozen=True)
class TermPoints:
    relations: tuple[tuple[NodePath, NodePath], ...]
    places: tuple[NodePath, ...]
    expressions: dict[NodePath, Expression]


def measure_points(checked: Checked) -> TermPoints:
    relations = []
    places = []
    expressions = {}

    def is_point(path: NodePath) -> bool:
        kind = checked.nodes[path].type
        return kind.space == "point" and kind.shape == (2,)

    def visit(node: Expression, path: NodePath, lone: bool) -> None:
        if path in checked.calls:
            _, node = checked.calls[path]
        if isinstance(node, CallValue) and node.function == "sub":
            first, second = (*path, "args", 0), (*path, "args", 1)
            if is_point(first) and is_point(second):
                relations.append((first, second))
                expressions[first], expressions[second] = node.args
                lone = False
        if lone and is_point(path):
            places.append(path)
            expressions[path] = node
            lone = False
        for relative, operand in children(node):
            visit(operand, (*path, *relative), lone)

    visit(checked.tree, (), True)
    return TermPoints(tuple(relations), tuple(places), expressions)
