from __future__ import annotations

import ast
from dataclasses import dataclass
from pathlib import Path

import pytest

from feedbax.testing.ast_scan import (
    SiteVisitor,
    StructuralSite,
    expr_string,
    scan_domain,
    scan_file,
    scan_tree,
    target_label,
)


@dataclass(frozen=True)
class CallSite(StructuralSite):
    target: str

    @property
    def structural_key(self) -> tuple[str, str, str]:
        return (*super().structural_key, self.target)


class CallVisitor(SiteVisitor[CallSite]):
    def visit_Call(self, node: ast.Call) -> None:
        self.sites.append(
            CallSite(self.relpath, self.qualname, node.lineno, target_label(node.func))
        )
        self.generic_visit(node)


def test_scan_file_tracks_nested_scope_and_stable_key(tmp_path: Path) -> None:
    source = tmp_path / "pkg" / "sample.py"
    source.parent.mkdir()
    source.write_text(
        "class Runner:\n"
        "    def run(self):\n"
        "        sink.write('x')\n",
        encoding="utf-8",
    )

    sites = scan_file(source, root=tmp_path, visitor_factory=CallVisitor)

    assert sites == [CallSite("pkg/sample.py", "Runner.run", 3, "sink.write")]
    assert sites[0].structural_key == ("pkg/sample.py", "Runner.run", "sink.write")


def test_scan_tree_is_deterministic_and_supports_exclusion(tmp_path: Path) -> None:
    (tmp_path / "b.py").write_text("b()\n", encoding="utf-8")
    (tmp_path / "a.py").write_text("a()\n", encoding="utf-8")
    (tmp_path / "ignored.py").write_text("ignored()\n", encoding="utf-8")

    sites = scan_tree(
        tmp_path,
        visitor_factory=CallVisitor,
        exclude=lambda path: path.name == "ignored.py",
    )

    assert [site.relpath for site in sites] == ["a.py", "b.py"]


def test_scan_domain_accepts_explicit_noncontiguous_files(tmp_path: Path) -> None:
    first = tmp_path / "one.py"
    second = tmp_path / "nested" / "two.py"
    second.parent.mkdir()
    first.write_text("one()\n", encoding="utf-8")
    second.write_text("two()\n", encoding="utf-8")

    sites = scan_domain([second, first], root=tmp_path, visitor_factory=CallVisitor)

    assert [site.target for site in sites] == ["two", "one"]


def test_scan_file_rejects_path_outside_explicit_root(tmp_path: Path) -> None:
    root = tmp_path / "root"
    root.mkdir()
    outside = tmp_path / "outside.py"
    outside.write_text("call()\n", encoding="utf-8")

    with pytest.raises(ValueError):
        scan_file(outside, root=root, visitor_factory=CallVisitor)


def test_expression_helpers_share_canonical_rendering() -> None:
    node = ast.parse("output_dir / name", mode="eval").body

    assert expr_string(node) == "output_dir / name"
    assert target_label(node) == expr_string(node)
    assert expr_string(None) == "<none>"
