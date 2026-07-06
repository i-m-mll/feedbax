from __future__ import annotations

import ast
from pathlib import Path

import pytest

pytestmark = [pytest.mark.feedbax_contract, pytest.mark.no_silent_substitution_contract]

REPO_ROOT = Path(__file__).resolve().parents[1]


def _function_node(path: str, name: str) -> ast.AST:
    tree = ast.parse((REPO_ROOT / path).read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == name:
            return node
    raise AssertionError(f"Function {name!r} not found in {path}")


def _class_method_node(path: str, class_name: str, method_name: str) -> ast.AST:
    tree = ast.parse((REPO_ROOT / path).read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef) and node.name == class_name:
            for child in node.body:
                if isinstance(child, ast.FunctionDef) and child.name == method_name:
                    return child
    raise AssertionError(f"Method {class_name}.{method_name} not found in {path}")


def _is_zero_constant(node: ast.AST) -> bool:
    return isinstance(node, ast.Constant) and node.value in (0, 0.0)


def test_acausal_assembly_does_not_default_missing_through_vars_to_zero() -> None:
    node = _function_node("feedbax/acausal/assembly.py", "_make_vector_field")

    offenders = []
    for call in (item for item in ast.walk(node) if isinstance(item, ast.Call)):
        if (
            isinstance(call.func, ast.Attribute)
            and call.func.attr == "get"
            and len(call.args) >= 2
            and _is_zero_constant(call.args[1])
        ):
            offenders.append(call.lineno)

    assert offenders == []


def test_config_tree_does_not_synthesize_placeholder_ldict_key() -> None:
    node = _function_node("feedbax/config/tree.py", "_expand_missing_levels")

    offenders = []
    for assignment in (
        item for item in ast.walk(node) if isinstance(item, (ast.Assign, ast.AnnAssign))
    ):
        value = assignment.value
        if (
            isinstance(value, ast.List)
            and len(value.elts) == 1
            and isinstance(value.elts[0], ast.Constant)
            and value.elts[0].value == "_"
        ):
            offenders.append(assignment.lineno)

    assert offenders == []


def test_loss_evaluation_does_not_default_missing_target_to_zero() -> None:
    node = _function_node("feedbax/runtime/retained_observables.py", "evaluate_loss_term")

    offenders = []
    for assignment in (
        item for item in ast.walk(node) if isinstance(item, (ast.Assign, ast.AnnAssign))
    ):
        targets = assignment.targets if isinstance(assignment, ast.Assign) else [assignment.target]
        if any(isinstance(target, ast.Name) and target.id == "target" for target in targets):
            if _is_zero_constant(assignment.value):
                offenders.append(assignment.lineno)

    assert offenders == []


def test_migration_registry_does_not_treat_versionless_as_current_by_default() -> None:
    node = _class_method_node(
        "feedbax/contracts/migrations.py",
        "SpecSchemaRegistry",
        "migrate",
    )

    offenders = []
    for assignment in (
        item for item in ast.walk(node) if isinstance(item, (ast.Assign, ast.AnnAssign))
    ):
        targets = assignment.targets if isinstance(assignment, ast.Assign) else [assignment.target]
        if not any(isinstance(target, ast.Name) and target.id == "resolved_source" for target in targets):
            continue
        for bool_op in (item for item in ast.walk(assignment.value) if isinstance(item, ast.BoolOp)):
            if isinstance(bool_op.op, ast.Or):
                for value in bool_op.values:
                    if isinstance(value, ast.Attribute) and value.attr == "current_version":
                        offenders.append(bool_op.lineno)

    assert offenders == []


def test_render_format_dispatch_does_not_passthrough_unknown_formats() -> None:
    node = _function_node("feedbax/plot/io.py", "_render_extension")

    offenders = []
    for call in (item for item in ast.walk(node) if isinstance(item, ast.Call)):
        if isinstance(call.func, ast.Attribute) and call.func.attr == "get":
            offenders.append(call.lineno)

    assert offenders == []
