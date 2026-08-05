"""Structural gate: every JSON parse in ``feedbax/contracts`` is accounted for.

``feedbax/contracts`` is the layer that reads authority documents — manifests,
packets, envelopes, compile locks, content-pinned bases, declarations, array
stores. Every one of those parses must go through
:func:`feedbax.contracts.strict_json.strict_json_loads`, which refuses a
document that states one member twice.

A parse that is genuinely trusted-internal — bytes this process serialized
moments earlier from a value it already holds — is not an authority boundary and
does not need the strict loader. Those are enumerated below, one entry per
enclosing function, each with the reason it is not a boundary. The gate fails
when a JSON parse appears anywhere in the package that is neither routed through
the strict loader nor listed here, so a new bypass cannot be added silently.

The permissive parsers this gate looks for are the ones that all share the same
last-value-wins behavior for a repeated object member name:

* ``json.loads`` / ``json.load``;
* ``json.JSONDecoder(...).decode``;
* pydantic's ``Model.model_validate_json`` and ``TypeAdapter(...).validate_json``
  (measured: pydantic 2.12 keeps the last value exactly as ``json.loads`` does).

Scope note: this gate covers ``feedbax/contracts`` only, which is the audited
authority-document layer. Parsers elsewhere in the package (orchestration,
training, analysis, web) are outside it.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

pytestmark = [pytest.mark.feedbax_contract, pytest.mark.strict_json_boundary_contract]

REPO_ROOT = Path(__file__).resolve().parents[1]
BOUNDARY_PACKAGE = Path("feedbax/contracts")

#: The strict loader itself parses with ``json.loads``; that call *is* the
#: boundary implementation.
GATE_EXEMPT_MODULES = frozenset({"feedbax/contracts/strict_json.py"})

#: Trusted-internal parses, keyed by (module, enclosing qualified name). Each
#: value states why the bytes carry no external authority. Adding an entry here
#: is a deliberate, reviewable claim — not a way to silence the gate.
TRUSTED_INTERNAL_PARSES: dict[tuple[str, str], str] = {
    (
        "feedbax/contracts/manifest.py",
        "TrainingManifestMetadataProjectionCustody._validate_custody",
    ): (
        "Round-trip of the canonical serialization of ``self.values``, produced "
        "in this process one expression earlier. The document that carried "
        "``self.values`` was already admitted by ``load_manifest_bytes``."
    ),
    ("feedbax/contracts/manifest_packet.py", "_manifest_data_for_import"): (
        "Deep copy of a mapping already admitted by ``strict_json_loads`` in "
        "``_validate_packet_manifests``, via bytes ``json.dumps`` produced one "
        "expression earlier."
    ),
    ("feedbax/contracts/evaluation_states.py", "_encode_mixed_leaves"): (
        "Round-trip of ``_canonical_metadata_leaf_bytes`` output for an "
        "in-memory leaf; canonical serialization emits each member once."
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
                    callee.attr
                    if isinstance(callee, ast.Attribute)
                    else getattr(callee, "id", "")
                )
                if name == "JSONDecoder":
                    found.append((_qualified_name(stack), node.lineno, ast.unparse(node.func)))
        inner = stack + [node] if isinstance(
            node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)
        ) else stack
        for child in ast.iter_child_nodes(node):
            visit(child, inner)

    visit(tree, [])
    return found


def _boundary_modules() -> list[Path]:
    root = REPO_ROOT / BOUNDARY_PACKAGE
    return sorted(path for path in root.rglob("*.py") if "__pycache__" not in path.parts)


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


def test_no_unaccounted_permissive_json_parse_in_the_contracts_layer() -> None:
    """A JSON parse here is either the strict loader or a documented exception."""
    observed = _observed_parses()
    offenders = sorted(key for key in observed if key not in TRUSTED_INTERNAL_PARSES)
    detail = "\n".join(
        f"  {module}:{sorted(observed[(module, qualname)])} in {qualname}"
        for module, qualname in offenders
    )
    assert not offenders, (
        "Permissive JSON parsing reached an unaccounted site in "
        f"{BOUNDARY_PACKAGE.as_posix()}:\n{detail}\n"
        "This layer reads authority documents, where a repeated object member "
        "name states two authorities for one fact and the standard parse "
        "silently keeps the last. Route the call through "
        "feedbax.contracts.strict_json.strict_json_loads, or — if the bytes "
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
"""
    found = {qualname for qualname, _lineno, _expr in _permissive_parse_calls(ast.parse(source))}
    assert found == {
        "parses_with_json_loads",
        "parses_with_json_load",
        "parses_with_decoder",
        "Reads.parses_with_pydantic",
        "parses_with_type_adapter",
    }
