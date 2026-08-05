"""Ratchet on package-level import cycles in ``feedbax``.

This is deliberately *not* a strict layering rule. Most subpackages currently sit
in a single strongly connected component of the package import graph, so a target
layering rule cannot be enforced before the target architecture exists. What is
enforceable today is a monotone upper bound: the set of package pairs that
participate in an import cycle is recorded in a checked-in baseline, and this
module fails when a package pair that is not in the baseline starts closing a
cycle. The baseline may shrink freely — a lane that removes cycle edges keeps
passing, and can lock the win in the same commit with::

    uv run --no-sync python scripts/import_cycle_baseline.py --write

Analysis is static (:mod:`ast`), never import-time. See
``scripts/import_cycle_baseline.py`` for the exact classification rules for
module-scope, function-local, and ``TYPE_CHECKING`` imports.
"""

from __future__ import annotations

import ast
import textwrap
import warnings
from pathlib import Path

import pytest

from scripts import import_cycle_baseline as ratchet

REPO_ROOT = Path(__file__).resolve().parents[1]
BASELINE_PATH = REPO_ROOT / ratchet.DEFAULT_BASELINE_PATH


def test_ratchet_analyzer_is_the_one_in_this_checkout() -> None:
    """Guard against an editable install resolving ``scripts`` to another checkout."""
    analyzer_path = Path(ratchet.__file__).resolve()
    assert analyzer_path == REPO_ROOT / "scripts/import_cycle_baseline.py", (
        f"import-cycle analyzer resolved to {analyzer_path}, which is outside this "
        f"checkout ({REPO_ROOT}); run pytest from the checkout under test"
    )


def test_no_new_package_import_cycle_edges() -> None:
    baseline = ratchet.baseline_edges(ratchet.load_baseline(BASELINE_PATH))
    graph, _summary = ratchet.analyze(REPO_ROOT)

    added = ratchet.new_cycle_edges(graph, baseline)
    assert not added, ratchet.format_violations(graph, added)

    removed = ratchet.removed_cycle_edges(graph, baseline)
    if removed:
        warnings.warn(
            "The import-cycle baseline is looser than the code: "
            f"{len(removed)} baseline edge(s) no longer close a cycle "
            f"({', '.join(f'{source} -> {target}' for source, target in removed)}). "
            f"Lock the win with: {ratchet.REGENERATE_COMMAND}",
            stacklevel=1,
        )


def test_baseline_document_is_normalized() -> None:
    """The checked-in baseline must be exactly what the regenerator emits."""
    document = ratchet.load_baseline(BASELINE_PATH)
    edges = ratchet.baseline_edges(document)

    assert len(document["cycle_edges"]) == len(edges), "baseline contains duplicate edges"
    assert document["cycle_edges"] == [list(edge) for edge in sorted(edges)], (
        "baseline cycle_edges are not sorted; regenerate with: "
        f"{ratchet.REGENERATE_COMMAND}"
    )
    assert all(source != target for source, target in edges), "baseline contains a self edge"


def test_baseline_rejects_an_unknown_schema(tmp_path: Path) -> None:
    """Durable emitted specs fail closed on an unsupported schema identity."""
    path = tmp_path / "package_cycle_baseline.json"
    path.write_text('{"schema": "feedbax.import_cycle_baseline.v0", "cycle_edges": []}\n')

    with pytest.raises(ValueError, match="unsupported import-cycle baseline schema"):
        ratchet.load_baseline(path)


def _write_package(root: Path, name: str, modules: dict[str, str]) -> None:
    package = root / "feedbax" / name
    package.mkdir(parents=True, exist_ok=True)
    (package / "__init__.py").write_text("", encoding="utf-8")
    for module, source in modules.items():
        (package / f"{module}.py").write_text(textwrap.dedent(source), encoding="utf-8")


def _synthetic_repo(tmp_path: Path) -> Path:
    """A three-package toy repo: ``low <-> high`` cycle plus an acyclic ``leaf``."""
    root = tmp_path / "repo"
    (root / "feedbax").mkdir(parents=True)
    (root / "feedbax" / "__init__.py").write_text("", encoding="utf-8")
    _write_package(root, "low", {"api": "from feedbax.high import thing\nfrom feedbax.leaf import x\n"})
    _write_package(root, "high", {"thing": "from feedbax.low import api\n"})
    _write_package(root, "leaf", {"x": "value = 1\n"})
    return root


def test_ratchet_fires_on_a_new_cycle_edge(tmp_path: Path) -> None:
    root = _synthetic_repo(tmp_path)
    graph, summary = ratchet.analyze(root)
    baseline = set(graph.cycle_edges)
    assert baseline == {("high", "low"), ("low", "high")}
    assert summary["packages_in_largest_cycle"] == 2

    # A new back-edge pulls the previously acyclic ``leaf`` package into the cycle.
    (root / "feedbax" / "leaf" / "x.py").write_text(
        "from feedbax.low import api\n\nvalue = 1\n", encoding="utf-8"
    )
    graph, _ = ratchet.analyze(root)
    added = ratchet.new_cycle_edges(graph, baseline)

    assert added == [("leaf", "low"), ("low", "leaf")]
    message = ratchet.format_violations(graph, added)
    assert "NEW CYCLE EDGE  leaf -> low" in message
    assert "feedbax.leaf.x:1 imports feedbax.low.api [leaf -> low]" in message
    assert ratchet.REGENERATE_COMMAND in message


def test_ratchet_does_not_fire_when_a_cycle_edge_is_removed(tmp_path: Path) -> None:
    root = _synthetic_repo(tmp_path)
    graph, _ = ratchet.analyze(root)
    baseline = set(graph.cycle_edges)

    (root / "feedbax" / "high" / "thing.py").write_text("value = 1\n", encoding="utf-8")
    graph, _ = ratchet.analyze(root)

    assert ratchet.new_cycle_edges(graph, baseline) == []
    assert ratchet.removed_cycle_edges(graph, baseline) == [("high", "low"), ("low", "high")]
    assert graph.cycle_edges == ()


def test_function_local_and_type_checking_imports_are_not_cycle_edges(tmp_path: Path) -> None:
    root = tmp_path / "repo"
    (root / "feedbax").mkdir(parents=True)
    (root / "feedbax" / "__init__.py").write_text("", encoding="utf-8")
    _write_package(root, "low", {"api": "from feedbax.high import thing\n"})
    _write_package(
        root,
        "high",
        {
            "thing": """
                from typing import TYPE_CHECKING

                if TYPE_CHECKING:
                    from feedbax.low import api


                def call():
                    from feedbax.low import api

                    return api
                """
        },
    )

    graph, _ = ratchet.analyze(root)

    assert graph.cycle_edges == ()
    assert ("low", "high") in graph.package_edges
    assert ("high", "low") not in graph.package_edges


def test_class_body_and_import_error_guarded_imports_are_cycle_edges(tmp_path: Path) -> None:
    root = tmp_path / "repo"
    (root / "feedbax").mkdir(parents=True)
    (root / "feedbax" / "__init__.py").write_text("", encoding="utf-8")
    _write_package(root, "low", {"api": "from feedbax.high import thing\n"})
    _write_package(
        root,
        "high",
        {
            "thing": """
                try:
                    from feedbax.low import api
                except ImportError:
                    api = None
                """,
            "holder": """
                class Holder:
                    from feedbax.low import api
                """,
        },
    )

    graph, _ = ratchet.analyze(root)

    assert set(graph.cycle_edges) == {("high", "low"), ("low", "high")}
    witnesses = {edge.source_module for edge in graph.witnesses(("high", "low"))}
    assert witnesses == {"feedbax.high.thing", "feedbax.high.holder"}


def test_relative_imports_are_attributed_to_the_imported_package(tmp_path: Path) -> None:
    root = tmp_path / "repo"
    (root / "feedbax").mkdir(parents=True)
    (root / "feedbax" / "__init__.py").write_text("", encoding="utf-8")
    _write_package(root, "low", {"api": "from ..high.thing import value\n"})
    _write_package(root, "high", {"thing": "from ..low import api\n"})

    graph, _ = ratchet.analyze(root)

    assert set(graph.cycle_edges) == {("high", "low"), ("low", "high")}
    assert {edge.target_module for edge in graph.witnesses(("low", "high"))} == {
        "feedbax.high.thing"
    }


def test_reconcile_separates_module_scope_from_all_scope(tmp_path: Path) -> None:
    """``--reconcile`` must count deferred imports only in its all-scope figures."""
    root = tmp_path / "repo"
    (root / "feedbax").mkdir(parents=True)
    (root / "feedbax" / "__init__.py").write_text("", encoding="utf-8")
    _write_package(root, "low", {"api": "from feedbax.high import thing\n"})
    _write_package(
        root,
        "high",
        {
            "thing": """
                def call():
                    from feedbax.low import api

                    return api
                """
        },
    )

    figures = ratchet.reconcile(root)

    assert figures["ratchet_module_scope_import_statements"] == 1
    assert figures["audit_all_scope_import_statements"] == 2
    assert figures["audit_mutual_pairs_module_scope_no_root"] == 0
    assert figures["audit_mutual_pairs_all_scope_no_root"] == 1


def test_module_scope_visitor_skips_deferred_scopes() -> None:
    tree = ast.parse(
        textwrap.dedent(
            """
            import a

            if TYPE_CHECKING:
                import b
            else:
                import c

            def f():
                import d

            async def g():
                import e

            h = lambda: __import__("f")

            class K:
                import g_module
            """
        )
    )
    visitor = ratchet.ModuleScopeImportVisitor()
    visitor.visit(tree)

    assert [entry[2] for entry in visitor.imports] == ["a", "c", "g_module"]
