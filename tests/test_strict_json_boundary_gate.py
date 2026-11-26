"""Structural gate: every JSON parse in audited authority packages is accounted for.

The audited packages read authority documents — manifests, packets, envelopes,
compile locks, content-pinned bases, declarations, array stores, session rows,
venue replies, job inputs. Every parse of bytes that came from outside the
process goes through one of the helpers in :mod:`feedbax.contracts.strict_json`,
which refuse a document that states one member twice:

* ``strict_json_loads`` for plain JSON;
* ``strict_model_validate_json`` for a Pydantic model;
* ``strict_validate_json`` for a Pydantic ``TypeAdapter``.

A parse that does not route through a helper is enumerated below, one entry per
enclosing function, each with the reason it is not an authority boundary. The
gate fails when a JSON parse appears anywhere in the audited packages that is
neither routed through a helper nor listed here, so a new bypass cannot be
added silently.

The permissive parsers this gate looks for are the ones that all share the same
last-value-wins behavior for a repeated object member name:

* ``json.loads`` / ``json.load``;
* ``json.JSONDecoder(...).decode``;
* pydantic's ``Model.model_validate_json`` and ``TypeAdapter(...).validate_json``
  (measured: pydantic 2.12 keeps the last value exactly as ``json.loads`` does).

The audited packages are the contract, analysis, orchestration, and
training layers. Web transport decoding remains outside this gate.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

pytestmark = [pytest.mark.feedbax_contract, pytest.mark.strict_json_boundary_contract]

REPO_ROOT = Path(__file__).resolve().parents[1]
BOUNDARY_PACKAGES = tuple(
    Path(package)
    for package in (
        "feedbax/contracts",
        "feedbax/analysis",
        "feedbax/orchestration",
        "feedbax/training",
    )
)

#: The strict loader itself parses with ``json.loads``; that call *is* the
#: boundary implementation.
GATE_EXEMPT_MODULES = frozenset({"feedbax/contracts/strict_json.py"})

#: Parses that do not route through a strict helper, keyed by (module, enclosing
#: qualified name). Each value states why the bytes carry no external authority.
#: Adding an entry here is a deliberate, reviewable claim — not a way to silence
#: the gate.
TRUSTED_INTERNAL_PARSES: dict[tuple[str, str], str] = {
    ("feedbax/contracts/portable.py", "validate_authoring"): (
        "Validation consumes json.dumps of the in-memory authoring mapping; "
        "the serializer emits each member once."
    ),
    ("feedbax/contracts/portable.py", "validate_configuration"): (
        "Field validation consumes json.dumps of the current in-memory runtime value."
    ),
    ("feedbax/contracts/_request_codec.py", "_project_absent"): (
        "Validates, in Pydantic's JSON mode, json.dumps of the absent value this "
        "function just restored."
    ),
    ("feedbax/contracts/request_identity.py", "decode_python_task_selection"): (
        "The selection bytes go through strict_json_loads; the remaining parse "
        "validates, in Pydantic's JSON mode, json.dumps of the value this function "
        "just decoded."
    ),
    ("feedbax/contracts/request_identity.py", "decode_request"): (
        "Stored objects go through the strict reader; the remaining parse validates, "
        "in Pydantic's JSON mode, json.dumps of the spec this function just restored."
    ),
    ("feedbax/contracts/request_identity.py", "effective_binding_id"): (
        "Parses the bytes encode_request produced in this function from an "
        "authenticated request."
    ),
    ("feedbax/orchestration/runpod_venue_cutoff.py", "_own_pod_exists"): (
        "Standard-library source sent to the venue before feedbax is importable; "
        "the parse refuses a repeated member with its own object_pairs_hook."
    ),
}

_PERMISSIVE_JSON_ATTRS = frozenset({"loads", "load"})
_PERMISSIVE_VALIDATOR_ATTRS = frozenset({"model_validate_json", "validate_json", "parse_raw"})


def _qualified_name(stack: list[ast.AST]) -> str:
    parts = [
        node.name
        for node in stack
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
    ]
    return ".".join(parts) if parts else "<module>"


def _permissive_parse_calls(tree: ast.AST) -> list[tuple[str, int, str]]:
    """Return (qualified name, line number, called expression) per permissive parse."""
    found: list[tuple[str, int, str]] = []

    def visit(node: ast.AST, stack: list[ast.AST]) -> None:
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
            attr = node.func.attr
            value = node.func.value
            is_json_module = isinstance(value, ast.Name) and value.id == "json"
            if (attr in _PERMISSIVE_JSON_ATTRS and is_json_module) or (
                attr in _PERMISSIVE_VALIDATOR_ATTRS
            ):
                found.append((_qualified_name(stack), node.lineno, ast.unparse(node.func)))
            if attr == "decode" and isinstance(value, ast.Call):
                callee = value.func
                name = (
                    callee.attr if isinstance(callee, ast.Attribute) else getattr(callee, "id", "")
                )
                if name == "JSONDecoder":
                    found.append((_qualified_name(stack), node.lineno, ast.unparse(node.func)))
        inner = (
            stack + [node]
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
            else stack
        )
        for child in ast.iter_child_nodes(node):
            visit(child, inner)

    visit(tree, [])
    return found


def _boundary_modules() -> list[Path]:
    return sorted(
        path
        for package in BOUNDARY_PACKAGES
        for path in (REPO_ROOT / package).rglob("*.py")
        if "__pycache__" not in path.parts
    )


def _observed_parses() -> dict[tuple[str, str], list[int]]:
    observed: dict[tuple[str, str], list[int]] = {}
    for path in _boundary_modules():
        relative = path.relative_to(REPO_ROOT).as_posix()
        if relative in GATE_EXEMPT_MODULES:
            continue
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for qualname, lineno, _expression in _permissive_parse_calls(tree):
            observed.setdefault((relative, qualname), []).append(lineno)
    return observed


def test_no_unaccounted_permissive_json_parse_in_audited_packages() -> None:
    """A JSON parse here is either a strict helper or a documented exception."""
    observed = _observed_parses()
    offenders = sorted(key for key in observed if key not in TRUSTED_INTERNAL_PARSES)
    detail = "\n".join(
        f"  {module}:{sorted(observed[(module, qualname)])} in {qualname}"
        for module, qualname in offenders
    )
    assert not offenders, (
        "Permissive JSON parsing reached an unaccounted site in "
        f"{', '.join(package.as_posix() for package in BOUNDARY_PACKAGES)}:\n{detail}\n"
        "This layer reads authority documents, where a repeated object member "
        "name states two authorities for one fact and the standard parse "
        "silently keeps the last. Route the call through a helper in "
        "feedbax.contracts.strict_json (strict_json_loads, "
        "strict_model_validate_json, strict_validate_json), or — if the bytes "
        "were serialized by this process from a value it already holds — add "
        "the site to TRUSTED_INTERNAL_PARSES in this module with the reason."
    )


def test_every_documented_exception_still_exists() -> None:
    """A stale allowlist entry would hide a real bypass behind a dead key."""
    observed = _observed_parses()
    stale = sorted(key for key in TRUSTED_INTERNAL_PARSES if key not in observed)
    assert not stale, (
        "TRUSTED_INTERNAL_PARSES names sites that no longer parse JSON: "
        f"{stale}. Remove them so the allowlist keeps meaning what it says."
    )


def test_every_documented_exception_states_a_reason() -> None:
    for key, reason in TRUSTED_INTERNAL_PARSES.items():
        assert reason.strip(), f"TRUSTED_INTERNAL_PARSES[{key}] must state why it is trusted"


def test_the_gate_detects_each_permissive_parser_form() -> None:
    """The detector is proven against the exact forms it claims to cover."""
    source = """
import json
import pydantic

def parses_with_json_loads(raw):
    return json.loads(raw)

def parses_with_json_load(handle):
    return json.load(handle)

def parses_with_decoder(raw):
    return json.JSONDecoder().decode(raw)

class Reads:
    def parses_with_pydantic(self, raw):
        return SomeModel.model_validate_json(raw)

def parses_with_type_adapter(raw):
    return pydantic.TypeAdapter(dict).validate_json(raw)

def routes_through_the_strict_loader(raw):
    return strict_json_loads(raw, ref="ok")

def routes_through_the_strict_model_reader(raw):
    return strict_model_validate_json(SomeModel, raw)

def routes_through_the_strict_adapter_reader(raw):
    return strict_validate_json(pydantic.TypeAdapter(dict), raw)
"""
    found = {qualname for qualname, _lineno, _expr in _permissive_parse_calls(ast.parse(source))}
    assert found == {
        "parses_with_json_loads",
        "parses_with_json_load",
        "parses_with_decoder",
        "Reads.parses_with_pydantic",
        "parses_with_type_adapter",
    }
