from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import pytest

from feedbax.testing.allowlist import (
    AllowlistEntry,
    AllowlistError,
    JsonBaseline,
    Scope,
    TomlListBaseline,
    compare_ratchet,
    diff_allowlist,
    enforce_ratchet,
    load_scoped_entries,
    write_shrink_only,
)


@dataclass(frozen=True)
class Finding:
    path: str
    scope: str
    kind: str


def _diff(findings: list[Finding], entries: list[AllowlistEntry[tuple[str]]]):
    return diff_allowlist(
        findings,
        entries,
        site_key=lambda site: (site.kind,),
        site_location=lambda site: (site.path, site.scope),
    )


def test_scope_dispatch_and_pluggable_keys_report_both_diff_sides() -> None:
    findings = [
        Finding("src/a.py", "run", "write"),
        Finding("scripts/build.py", "<module>", "emit"),
        Finding("tests/test_a.py", "test_a", "unknown"),
    ]
    entries = [
        AllowlistEntry(Scope("python_scope", "src/a.py", "run"), "team", "owned", ("write",)),
        AllowlistEntry(Scope("glob", "scripts/*.py"), "team", "scripts", ("emit",)),
        AllowlistEntry(Scope("file", "src/dead.py"), "team", "legacy", None),
    ]

    result = _diff(findings, entries)

    assert result.unlisted == (findings[2],)
    assert result.dead_entries == (entries[2],)
    with pytest.raises(AllowlistError, match="unlisted findings.*dead allowlist entries"):
        result.require_clean()


@pytest.mark.parametrize("field", ["owner", "reason"])
def test_entry_requires_owner_and_reason(field: str) -> None:
    values = {"owner": "team", "reason": "exception"}
    values[field] = " "
    with pytest.raises(ValueError, match=field):
        AllowlistEntry(Scope("file", "src/a.py"), **values)


def test_toml_loader_accepts_reason_alias_and_all_scope_kinds(tmp_path: Path) -> None:
    path = tmp_path / "allowlist.toml"
    path.write_text(
        "[[python_scope]]\npath='src/a.py'\nqualname='run'\nowner='a'\nreason='why'\n"
        "kind='write'\n\n"
        "[[file]]\npath='src/b.py'\nowner='b'\nrationale='why'\nkind='emit'\n\n"
        "[[glob]]\npattern='tests/*.py'\nowner='c'\nreason='why'\nkind='test'\n",
        encoding="utf-8",
    )

    entries = load_scoped_entries(path, key_from_entry=lambda raw: (raw["kind"],))

    assert [entry.scope.kind for entry in entries] == ["python_scope", "file", "glob"]
    assert [entry.key for entry in entries] == [("write",), ("emit",), ("test",)]


def test_json_backend_enforces_and_writes_shrink_only(tmp_path: Path) -> None:
    path = tmp_path / "baseline.json"
    path.write_text(json.dumps(["a", "b"]), encoding="utf-8")
    backend = JsonBaseline(path)

    with pytest.raises(AllowlistError, match="ratchet mismatch"):
        enforce_ratchet(["a"], backend)
    with pytest.raises(AllowlistError, match="refusing ratchet growth"):
        write_shrink_only(["a", "b", "c"], backend)

    assert write_shrink_only(["a"], backend) == {"a"}
    assert json.loads(path.read_text(encoding="utf-8")) == ["a"]


def test_json_backend_can_seed_and_explicitly_grow(tmp_path: Path) -> None:
    backend = JsonBaseline(tmp_path / "nested" / "baseline.json")

    write_shrink_only(["seed"], backend)
    write_shrink_only(["seed", "new"], backend, allow_growth=True)

    assert backend.load() == {"seed", "new"}


def test_toml_list_backend_reads_array_of_tables_without_rewriting(tmp_path: Path) -> None:
    path = tmp_path / "ratchet.toml"
    path.write_text("[[sites]]\npath='a.py'\nkind='write'\n", encoding="utf-8")
    backend = TomlListBaseline(path, "sites", lambda raw: (raw["path"], raw["kind"]))

    assert backend.load() == {("a.py", "write")}
    assert compare_ratchet({("a.py", "write")}, backend.load()).is_clean
    with pytest.raises(NotImplementedError, match="comments"):
        backend.write(set())
