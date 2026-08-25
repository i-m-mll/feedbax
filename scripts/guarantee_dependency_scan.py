#!/usr/bin/env python
"""Answer "what actually depends on each guaranteed interface".

One deterministic program, one command, one machine-readable output file, one
record per dependency. A record names the guaranteed item, the file, the line or
JSON pointer, the channel that carries the dependency, and the loader or
dispatcher in this library that makes the dependency live.

The program decides nothing. It deletes nothing. It changes no guarantee. It is
evidence machinery, and every claim it makes is traceable to a byte offset in a
named file.

Run:

    uv run --no-sync python scripts/guarantee_dependency_scan.py

The result lands in ``_evidence/guarantee_dependency_scan.v1.json``, which is an
ignored path: the scan is evidence produced from tracked inputs, not a tracked
deliverable, and it is reproduced by rerunning the command.

Design notes that are load-bearing rather than incidental:

*   The guarantee set is read from the versioned policy manifest, which is the
    sole structured authority for the concrete plugin API, and cross-checked
    against the rendered table in the policy document. A row whose document cell
    defers to the manifest ("inventory below") must carry a manifest inventory,
    and a row that yields no public names at all is a hard failure. A previous
    sweep parsed only the document, resolved the two deferring rows to the empty
    list, searched for nothing, and reported them unconsumed. That failure mode
    is structurally impossible here: see ``GuaranteeSet.load``.

*   Source roots are enumerated from an explicit NUL-delimited tracked-file
    listing. Durable-artifact roots are declared separately and walked, because
    durable run output is deliberately untracked.

*   Artifact trees are never excluded wholesale. Sealed source snapshots inside
    them are excluded, and only those, because they are verbatim copies of the
    scanned source and manufacture false consumers. Snapshot roots are detected
    structurally by nested-checkout markers rather than by one hard-coded
    directory name, because in practice they appear under several names.

*   Every channel carries a positive control. The control corpus is scanned on
    every run, and a channel that returns zero on its own control is reported as
    ``broken`` and its zeros elsewhere are withheld.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import io
import json
import os
import pickletools
import re
import shutil
import subprocess
import sys
import tomllib
from collections.abc import Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

OUTPUT_SCHEMA_ID = "feedbax.guarantee_dependency_scan"
OUTPUT_SCHEMA_VERSION = "feedbax.guarantee_dependency_scan.v1"

DEFAULT_MANIFEST = (
    "external/feedbax_conformance_fixture/src/feedbax_external_conformance/policy_manifest.v1.json"
)
DEFAULT_POLICY_DOC = "docs/design/downstream_interface_stability.md"
DEFAULT_CORPORA = "scripts/guarantee_dependency_scan.corpora.json"
DEFAULT_OUTPUT = "_evidence/guarantee_dependency_scan.v1.json"

GUARANTEE_MARK_START = "<!-- policy-guarantees:start -->"
GUARANTEE_MARK_END = "<!-- policy-guarantees:end -->"

#: Root distribution packages whose namespaces this policy governs. A Python
#: symbol only counts as a dependency when it is bound by an import from one of
#: these, which is what keeps a downstream package's own identically spelled
#: constant from being mistaken for a guaranteed one.
GOVERNED_ROOT_PACKAGES = ("feedbax", "feedbax_external_conformance")

IDENTIFIER_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
DOTTED_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*(?:\.[A-Za-z_][A-Za-z0-9_]*)*$")
BACKTICK_RE = re.compile(r"`([^`]+)`")
SCHEMA_VERSION_TAIL_RE = re.compile(r"\.v\d+$")
CONSOLE_SCRIPT_RE = re.compile(r"^feedbax(?:-[a-z0-9]+)*$")
ENV_VAR_RE = re.compile(r"^FEEDBAX_[A-Z0-9_]+$")
OBJECT_REF_RE = re.compile(
    r"^(?P<module>[A-Za-z_][A-Za-z0-9_]*(?:\.[A-Za-z_][A-Za-z0-9_]*)*)"
    r"[:.](?P<attr>[A-Za-z_][A-Za-z0-9_]*)$"
)

#: Directory names never scanned in any corpus.
ALWAYS_SKIPPED_DIRS = frozenset(
    {
        ".git",
        ".venv",
        "venv",
        "node_modules",
        "__pycache__",
        "site-packages",
        ".mypy_cache",
        ".pytest_cache",
        ".ruff_cache",
        ".jax_cache",
        ".uv-cache",
        "dist",
        "build",
        ".eggs",
        ".tox",
        ".hypothesis",
    }
)

#: Directory names that are, by convention in this ecosystem, sealed copies of a
#: source tree captured next to run output. Presence of a nested-checkout marker
#: is the primary structural test; these names are the belt to that braces.
SNAPSHOT_DIR_NAMES = frozenset(
    {".repo-snapshots", "repo-snapshots", "provider-snapshots", "source-snapshots"}
)

#: Files whose presence marks a directory as a nested checkout copy. A durable
#: artifact tree does not contain a package build definition; a sealed copy of a
#: repository always does.
CHECKOUT_MARKERS = ("pyproject.toml", "setup.py", "setup.cfg")

#: Document keys whose values are free text by contract. A guaranteed name found
#: inside one of these is prose, not a dependency.
PROSE_KEYS = frozenset(
    {
        "note",
        "notes",
        "comment",
        "comments",
        "description",
        "summary",
        "reason",
        "rationale",
        "message",
        "detail",
        "details",
        "text",
        "body",
        "docstring",
        "help",
        "title",
        "caption",
        "explanation",
        "commentary",
        "prose",
        "log",
    }
)

TEXT_DOCUMENT_SUFFIXES = frozenset({".json", ".jsonl", ".yaml", ".yml", ".toml"})
PYTHON_SUFFIXES = frozenset({".py", ".pyi"})
PICKLE_SUFFIXES = frozenset({".pkl", ".pickle", ".eqx", ".ckpt"})
PLAIN_TEXT_SUFFIXES = frozenset(
    {".md", ".txt", ".sh", ".cfg", ".ini", ".env", ".rst", ".qmd", ".ts", ".tsx", ".js"}
)

CHANNELS = (
    "python-import",
    "python-attribute",
    "python-star-import",
    "schema-id-string",
    "schema-family-prefix",
    "document-type-discriminator",
    "document-field-name",
    "registry-identifier",
    "plugin-family",
    "entry-point",
    "dynamic-import-path",
    "pickle-class-identity",
    "pytree-type-fingerprint",
    "console-script",
    "environment-variable",
    "binary-store-identity",
)


class ScanError(RuntimeError):
    """A condition the program refuses to work around."""


# --------------------------------------------------------------------------
# Guarantee set
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class GuaranteeItem:
    """One concrete guaranteed thing a downstream consumer can depend on."""

    row_id: str
    kind: str
    value: str
    origin: str
    detail: Mapping[str, Any] = field(default_factory=dict)

    def sort_key(self) -> tuple[str, str, str]:
        return (self.row_id, self.kind, self.value)


@dataclass(frozen=True)
class GuaranteeRow:
    row_id: str
    namespaces: tuple[str, ...]
    public_names: tuple[str, ...]
    schema_ids: tuple[str, ...]
    case_ids: tuple[str, ...]
    coverage_status: str
    names_origin: str
    doc_prose: str


@dataclass
class GuaranteeSet:
    rows: tuple[GuaranteeRow, ...]
    items: tuple[GuaranteeItem, ...]
    manifest_path: str
    doc_path: str
    manifest_sha256: str
    doc_sha256: str
    crosscheck: dict[str, Any]

    @property
    def row_ids(self) -> tuple[str, ...]:
        return tuple(row.row_id for row in self.rows)

    def rows_by_id(self) -> dict[str, GuaranteeRow]:
        return {row.row_id: row for row in self.rows}

    # -- loading ---------------------------------------------------------

    @classmethod
    def load(cls, manifest_path: Path, doc_path: Path) -> GuaranteeSet:
        manifest_bytes = manifest_path.read_bytes()
        doc_bytes = doc_path.read_bytes()
        manifest = json.loads(manifest_bytes.decode("utf-8"))
        doc_rows = _parse_policy_table(doc_bytes.decode("utf-8"))

        manifest_rows = manifest.get("guaranteed_rows")
        if not isinstance(manifest_rows, list) or not manifest_rows:
            raise ScanError(f"{manifest_path}: guaranteed_rows is missing or empty")

        manifest_by_id: dict[str, Mapping[str, Any]] = {}
        for entry in manifest_rows:
            row_id = entry.get("row_id")
            if not row_id:
                raise ScanError(f"{manifest_path}: a guaranteed row has no row_id")
            if row_id in manifest_by_id:
                raise ScanError(f"{manifest_path}: duplicate row_id {row_id!r}")
            manifest_by_id[row_id] = entry

        doc_by_id: dict[str, Mapping[str, str]] = {}
        for cells in doc_rows:
            row_id = _strip_ticks(cells["Row ID"])
            if row_id in doc_by_id:
                raise ScanError(f"{doc_path}: duplicate row_id {row_id!r} in policy table")
            doc_by_id[row_id] = cells

        problems: list[str] = []
        if set(manifest_by_id) != set(doc_by_id):
            only_manifest = sorted(set(manifest_by_id) - set(doc_by_id))
            only_doc = sorted(set(doc_by_id) - set(manifest_by_id))
            problems.append(
                "manifest and policy document disagree on the guaranteed row set: "
                f"manifest-only={only_manifest} document-only={only_doc}"
            )

        input_row_count = len(manifest_by_id)
        rows: list[GuaranteeRow] = []
        items: list[GuaranteeItem] = []
        soft_notes: list[str] = []

        for row_id in sorted(manifest_by_id):
            m_row = manifest_by_id[row_id]
            d_row = doc_by_id.get(row_id)
            if d_row is None:
                continue

            doc_namespaces = _backticked(d_row["Stable namespace"], DOTTED_RE)
            doc_names = _backticked(d_row["Stable public names"], IDENTIFIER_RE)
            doc_cases = _backticked(d_row["External case IDs"], IDENTIFIER_RE)
            doc_prose = d_row["Durable schemas and behavior"]

            inventory = m_row.get("plugin_api") or m_row.get("public_api") or {}
            inv_namespaces: list[str] = []
            inv_names: list[str] = []
            for block in inventory.get("namespaces", ()):  # type: ignore[union-attr]
                namespace = block.get("namespace")
                names = block.get("public_names") or []
                if not namespace:
                    problems.append(f"{row_id}: manifest inventory block has no namespace")
                    continue
                if not names:
                    problems.append(
                        f"{row_id}: manifest inventory namespace {namespace!r} lists no public names"
                    )
                inv_namespaces.append(namespace)
                inv_names.extend(names)

            defers_to_manifest = "inventory below" in d_row["Stable public names"].lower()
            if defers_to_manifest and not inv_names:
                problems.append(
                    f"{row_id}: the policy document defers its public-name inventory to the "
                    "manifest, and the manifest carries none. Searching for the empty list "
                    "would report this row unconsumed without looking at anything."
                )
            if inv_names and not defers_to_manifest and doc_names:
                doc_only = sorted(set(doc_names) - set(inv_names))
                inv_only = sorted(set(inv_names) - set(doc_names))
                if doc_only or inv_only:
                    problems.append(
                        f"{row_id}: manifest inventory and document name list disagree "
                        f"(document-only={doc_only} manifest-only={inv_only})"
                    )

            namespaces = _dedupe(list(doc_namespaces) + inv_namespaces)
            public_names = _dedupe(list(doc_names) + inv_names)

            if not public_names:
                problems.append(
                    f"{row_id}: resolved to zero public names from both the manifest and the "
                    "policy document. A row with nothing to search for cannot be reported "
                    "unconsumed."
                )
            unexpected_ns = sorted(set(inv_namespaces) - set(doc_namespaces))
            if unexpected_ns:
                problems.append(
                    f"{row_id}: manifest declares namespaces the document does not: {unexpected_ns}"
                )

            manifest_cases = tuple(m_row.get("case_ids") or ())
            if tuple(doc_cases) != manifest_cases:
                problems.append(
                    f"{row_id}: external case IDs disagree "
                    f"(document={list(doc_cases)} manifest={list(manifest_cases)})"
                )
            coverage = m_row.get("coverage_status") or ""
            if bool(manifest_cases) != (coverage == "covered"):
                problems.append(
                    f"{row_id}: coverage_status {coverage!r} disagrees with case IDs "
                    f"{list(manifest_cases)}"
                )

            schemas = m_row.get("schemas") or {}
            schema_ids: list[str] = []
            for bucket in ("current", "migrated", "rejected"):
                for value in schemas.get(bucket, ()):  # type: ignore[union-attr]
                    if value in {"unknown", "unversioned"}:
                        continue
                    schema_ids.append(value)
                    items.append(
                        GuaranteeItem(
                            row_id=row_id,
                            kind="schema_id",
                            value=value,
                            origin=f"manifest.schemas.{bucket}",
                            detail={"bucket": bucket},
                        )
                    )
            for value in schemas.get("current", ()):  # type: ignore[union-attr]
                if value not in {"unknown", "unversioned"} and value not in doc_prose:
                    soft_notes.append(
                        f"{row_id}: manifest current schema {value!r} is not restated verbatim "
                        "in the document's durable-schema cell"
                    )

            # Version-stripped schema families, so older live versions are found.
            for value in _dedupe(schema_ids):
                family = SCHEMA_VERSION_TAIL_RE.sub("", value)
                if family != value and "." in family:
                    items.append(
                        GuaranteeItem(
                            row_id=row_id,
                            kind="schema_family",
                            value=family,
                            origin="derived:version-stripped",
                            detail={"from_schema_id": value},
                        )
                    )
                    # A schema family tail is also the durable field name that
                    # implies the type without naming it.
                    tail = family.rsplit(".", 1)[-1]
                    # A single generic word ("figure", "report", "graph") is not a
                    # field name that implies a type; it is a word. Only a
                    # multi-word snake_case tail is specific enough to be evidence.
                    if IDENTIFIER_RE.match(tail) and "_" in tail and len(tail) >= 8:
                        items.append(
                            GuaranteeItem(
                                row_id=row_id,
                                kind="field_name",
                                value=tail,
                                origin="derived:schema-family-tail",
                                detail={"from_schema_family": family},
                            )
                        )

            # Schema identities that the document states but the manifest does
            # not enumerate (prose-only rows still have durable identities).
            for token in _backticked(doc_prose, DOTTED_RE):
                if token.startswith("feedbax.") and SCHEMA_VERSION_TAIL_RE.search(token):
                    if token not in schema_ids:
                        schema_ids.append(token)
                        items.append(
                            GuaranteeItem(
                                row_id=row_id,
                                kind="schema_id",
                                value=token,
                                origin="document:durable-schema-cell",
                                detail={"bucket": "document"},
                            )
                        )
                        family = SCHEMA_VERSION_TAIL_RE.sub("", token)
                        items.append(
                            GuaranteeItem(
                                row_id=row_id,
                                kind="schema_family",
                                value=family,
                                origin="derived:version-stripped",
                                detail={"from_schema_id": token},
                            )
                        )

            for name in public_names:
                items.append(
                    GuaranteeItem(
                        row_id=row_id,
                        kind="public_name",
                        value=name,
                        origin="manifest-inventory" if name in inv_names else "document-table",
                        detail={"namespaces": list(namespaces)},
                    )
                )
            for namespace in namespaces:
                items.append(
                    GuaranteeItem(
                        row_id=row_id,
                        kind="namespace",
                        value=namespace,
                        origin="manifest-inventory"
                        if namespace in inv_namespaces
                        else "document-table",
                    )
                )
            for case_id in manifest_cases:
                items.append(
                    GuaranteeItem(
                        row_id=row_id, kind="case_id", value=case_id, origin="manifest.case_ids"
                    )
                )

            items.extend(_plugin_family_items(row_id, inventory))
            items.extend(_cli_items(row_id, inventory, d_row))

            rows.append(
                GuaranteeRow(
                    row_id=row_id,
                    namespaces=tuple(namespaces),
                    public_names=tuple(public_names),
                    schema_ids=tuple(_dedupe(schema_ids)),
                    case_ids=manifest_cases,
                    coverage_status=coverage,
                    names_origin="manifest-inventory" if defers_to_manifest else "document-table",
                    doc_prose=doc_prose,
                )
            )

        if len(rows) != input_row_count:
            problems.append(
                "namespace normalization changed the row count: "
                f"{input_row_count} input rows produced {len(rows)} normalized rows"
            )

        if problems:
            raise ScanError(
                "guarantee set is not trustworthy; refusing to search:\n  - "
                + "\n  - ".join(problems)
            )

        return cls(
            rows=tuple(rows),
            items=tuple(sorted(_dedupe_items(items), key=GuaranteeItem.sort_key)),
            manifest_path=str(manifest_path),
            doc_path=str(doc_path),
            manifest_sha256=hashlib.sha256(manifest_bytes).hexdigest(),
            doc_sha256=hashlib.sha256(doc_bytes).hexdigest(),
            crosscheck={
                "row_count": len(rows),
                "status": "agree",
                "soft_notes": sorted(set(soft_notes)),
            },
        )


def _plugin_family_items(row_id: str, inventory: Mapping[str, Any]) -> list[GuaranteeItem]:
    """Family, registration, and callback names from the structured plugin API."""
    items: list[GuaranteeItem] = []
    for family in inventory.get("families", ()) or ():
        key = family.get("key")
        if key:
            items.append(
                GuaranteeItem(
                    row_id=row_id,
                    kind="plugin_family_key",
                    value=key,
                    origin="manifest.plugin_api.families",
                )
            )
        for bucket in ("registry_methods",):
            for value in family.get(bucket, ()) or ():
                items.append(
                    GuaranteeItem(
                        row_id=row_id,
                        kind="registry_method",
                        value=value,
                        origin=f"manifest.plugin_api.families.{bucket}",
                        detail={"family": key},
                    )
                )
        for bucket in ("registry_type", "callback_types", "support_types"):
            values = family.get(bucket)
            if isinstance(values, str):
                values = [values]
            for value in values or ():
                items.append(
                    GuaranteeItem(
                        row_id=row_id,
                        kind="public_name",
                        value=value,
                        origin=f"manifest.plugin_api.families.{bucket}",
                        detail={"family": key},
                    )
                )
        for value in family.get("public_consumers", ()) or ():
            items.append(
                GuaranteeItem(
                    row_id=row_id,
                    kind="object_reference",
                    value=value,
                    origin="manifest.plugin_api.families.public_consumers",
                    detail={"family": key},
                )
            )
    for value in inventory.get("direct_entrypoint_imports", ()) or ():
        items.append(
            GuaranteeItem(
                row_id=row_id,
                kind="public_name",
                value=value,
                origin="manifest.plugin_api.direct_entrypoint_imports",
            )
        )
    return items


def _cli_items(
    row_id: str, inventory: Mapping[str, Any], doc_row: Mapping[str, str]
) -> list[GuaranteeItem]:
    items: list[GuaranteeItem] = []
    for value in inventory.get("cli", ()) or ():
        items.append(
            GuaranteeItem(
                row_id=row_id, kind="cli_invocation", value=value, origin="manifest.public_api.cli"
            )
        )
    haystack = " ".join(doc_row.values())
    for token in BACKTICK_RE.findall(haystack):
        head = token.split()[0] if token.split() else ""
        if CONSOLE_SCRIPT_RE.match(head) and head != "feedbax":
            items.append(
                GuaranteeItem(
                    row_id=row_id,
                    kind="console_script",
                    value=head,
                    origin="document-table",
                    detail={"invocation": token},
                )
            )
            if token != head:
                items.append(
                    GuaranteeItem(
                        row_id=row_id, kind="cli_invocation", value=token, origin="document-table"
                    )
                )
    return items


def _parse_policy_table(text: str) -> list[dict[str, str]]:
    if GUARANTEE_MARK_START not in text or GUARANTEE_MARK_END not in text:
        raise ScanError("policy document has no policy-guarantees:start/end markers")
    body = text.split(GUARANTEE_MARK_START, 1)[1].split(GUARANTEE_MARK_END, 1)[0]
    lines = [line for line in body.strip().splitlines() if line.strip().startswith("|")]
    if len(lines) < 3:
        raise ScanError("policy-guarantees table has no rows")
    header = [cell.strip() for cell in lines[0].strip().strip("|").split("|")]
    expected = [
        "Row ID",
        "Stable namespace",
        "Stable public names",
        "Durable schemas and behavior",
        "External case IDs",
    ]
    if header != expected:
        raise ScanError(f"policy-guarantees table header changed: {header!r} != {expected!r}")
    rows: list[dict[str, str]] = []
    for line in lines[2:]:
        cells = [cell.strip() for cell in line.strip().strip("|").split("|")]
        if len(cells) != len(header):
            raise ScanError(
                f"policy-guarantees row has {len(cells)} cells, expected {len(header)}: "
                f"{cells[0][:60]!r}"
            )
        rows.append(dict(zip(header, cells, strict=True)))
    return rows


def _backticked(cell: str, pattern: re.Pattern[str]) -> list[str]:
    return _dedupe([token for token in BACKTICK_RE.findall(cell) if pattern.match(token)])


def _strip_ticks(value: str) -> str:
    return value.strip().strip("`").strip()


def _dedupe(values: Iterable[str]) -> list[str]:
    seen: dict[str, None] = {}
    for value in values:
        seen.setdefault(value, None)
    return list(seen)


def _dedupe_items(items: Iterable[GuaranteeItem]) -> list[GuaranteeItem]:
    seen: dict[tuple[str, str, str], GuaranteeItem] = {}
    for item in items:
        seen.setdefault((item.row_id, item.kind, item.value), item)
    return list(seen.values())


# --------------------------------------------------------------------------
# Runtime facts: query this library's own structures rather than infer them
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class RuntimeSource:
    """One live library structure that owns the truth for a dependency channel."""

    source_id: str
    module: str
    expression: str
    item_kind: str
    row_ids: tuple[str, ...]
    note: str


#: Where a channel's truth lives inside the library. These are evaluated, not
#: read textually, so that a rename in the library is visible as a scan failure
#: rather than as silently missing evidence.
RUNTIME_SOURCES: tuple[RuntimeSource, ...] = (
    RuntimeSource(
        "manifest-kinds",
        "feedbax.contracts.manifest",
        "sorted(MANIFEST_MODELS)",
        "manifest_kind",
        (),
        "kind-to-model dispatch map used by manifest loading",
    ),
    RuntimeSource(
        "manifest-spec-payload-fields",
        "feedbax.contracts.manifest",
        "sorted({f for fields in SPEC_PAYLOAD_FIELDS_BY_MANIFEST_KIND.values() for f in fields})",
        "field_name",
        (),
        "nested field names that imply a spec type without naming it",
    ),
    RuntimeSource(
        "manifest-kind-directories",
        "feedbax.contracts.manifest",
        "sorted(MANIFEST_KIND_DIRECTORIES.values())",
        "storage_directory",
        (),
        "authoritative on-disk layout for every persisted manifest kind",
    ),
    RuntimeSource(
        "spec-schema-families",
        "feedbax.contracts.migrations",
        "sorted({family.identity for family in default_spec_registry.families()})",
        "schema_family",
        (),
        "registered structured-spec families; row attribution is by prefix match",
    ),
    RuntimeSource(
        "spec-current-versions",
        "feedbax.contracts.migrations",
        "sorted({family.current_version for family in default_spec_registry.families()})",
        "schema_id",
        (),
        "registered current schema versions; row attribution is by prefix match",
    ),
    RuntimeSource(
        "component-type-ids",
        "feedbax.component_registry",
        "sorted(ComponentRegistry().names())",
        "registry_identifier",
        ("component-registration", "graph-spec"),
        "component type ids resolved dynamically from the component registry",
    ),
    RuntimeSource(
        "report-recipe-types",
        "feedbax.analysis.reports",
        "sorted(_report_recipe_keys())",
        "registry_identifier",
        ("report-surface",),
        "report type keys resolved dynamically from the report recipe registry",
    ),
    RuntimeSource(
        "driver-names",
        "feedbax.orchestration.drivers.builtins",
        "sorted(build_builtin_driver_registry().registered_names())",
        "registry_identifier",
        ("orchestration-driver",),
        "driver names resolved dynamically from the injected driver registry",
    ),
    RuntimeSource(
        "application-registry-keys",
        "feedbax.plugins.application",
        "sorted({key.family for key in APPLICATION_REGISTRY_KEYS})",
        "plugin_family_key",
        ("plugin-bootstrap",),
        "plugin bootstrap registry family table",
    ),
)


def collect_runtime_facts(library_root: Path) -> dict[str, Any]:
    """Evaluate the library's own structures. A failure here is reported, never guessed."""
    facts: dict[str, Any] = {"available": False, "sources": {}, "errors": []}
    sys_path_added = False
    if str(library_root) not in sys.path:
        sys.path.insert(0, str(library_root))
        sys_path_added = True
    try:
        for source in RUNTIME_SOURCES:
            try:
                module = __import__(source.module, fromlist=["*"])
                namespace = {k: getattr(module, k) for k in dir(module) if not k.startswith("__")}
                namespace["_report_recipe_keys"] = _make_report_recipe_keys(module)
                values = eval(
                    source.expression, {"__builtins__": {"sorted": sorted, "set": set}}, namespace
                )  # noqa: S307
                values = [str(v) for v in values]
                if not values:
                    facts["errors"].append(
                        f"{source.source_id}: {source.module}:{source.expression} returned nothing"
                    )
                facts["sources"][source.source_id] = {
                    "module": source.module,
                    "expression": source.expression,
                    "item_kind": source.item_kind,
                    "row_ids": list(source.row_ids),
                    "note": source.note,
                    "values": values,
                }
            except Exception as exc:  # noqa: BLE001 - reported, not swallowed
                facts["errors"].append(f"{source.source_id}: {type(exc).__name__}: {exc}")
        facts["available"] = bool(facts["sources"])
    finally:
        if sys_path_added:
            sys.path.remove(str(library_root))
    return facts


def _make_report_recipe_keys(module: Any):
    def _keys() -> list[str]:
        registry = module.ReportRecipeRegistry()
        module.register_builtin_report_recipes(registry)
        return sorted(registry.keys())

    return _keys


#: Row id used when a live library value cannot be attributed to one guaranteed
#: row. These records are kept and reported, but they never credit a row with
#: consumption it cannot be shown to have.
UNATTRIBUTED_ROW = "__unattributed__"


def attribute_value(
    value: str, guarantees: GuaranteeSet, declared: Sequence[str]
) -> tuple[str, ...]:
    """Decide which guaranteed rows a live library value belongs to.

    Attribution walks from specific to general and stops at the first level that
    matches. Crediting several rows for one value would let an unrelated row read
    as consumed, which is the expensive direction of error here.
    """
    by_name = tuple(row.row_id for row in guarantees.rows if value in row.public_names)
    if by_name:
        return by_name
    family = SCHEMA_VERSION_TAIL_RE.sub("", value)
    by_schema = tuple(
        row.row_id
        for row in guarantees.rows
        if value in row.schema_ids
        or any(SCHEMA_VERSION_TAIL_RE.sub("", s) == family for s in row.schema_ids)
    )
    if by_schema:
        return by_schema
    stem = value[: -len("_spec")] if value.endswith("_spec") else value
    for candidate in (f"feedbax.spec.{value}", f"feedbax.spec.{stem}"):
        by_derived = tuple(
            row.row_id
            for row in guarantees.rows
            if any(SCHEMA_VERSION_TAIL_RE.sub("", s) == candidate for s in row.schema_ids)
        )
        if by_derived:
            return by_derived
    if declared:
        return tuple(declared)
    return (UNATTRIBUTED_ROW,)


def runtime_items(
    facts: Mapping[str, Any], guarantees: GuaranteeSet, *, include_ungoverned: bool = False
) -> list[GuaranteeItem]:
    """Turn live library structures into guaranteed items attributed to rows.

    A live value that belongs to no guaranteed row describes ungoverned surface.
    It is real, and downstream may well depend on it, but it answers a different
    question than this program asks, so it is searched only on request.
    """
    items: list[GuaranteeItem] = []
    known = set(guarantees.row_ids) | ({UNATTRIBUTED_ROW} if include_ungoverned else set())

    for source_id, payload in sorted(facts.get("sources", {}).items()):
        declared_rows = tuple(payload["row_ids"])
        kind = payload["item_kind"]
        for value in payload["values"]:
            for row_id in attribute_value(value, guarantees, declared_rows):
                if row_id not in known:
                    continue
                items.append(
                    GuaranteeItem(
                        row_id=row_id,
                        kind=kind,
                        value=value,
                        origin=f"runtime:{source_id}",
                        detail={"source": source_id},
                    )
                )
    return items


def console_script_items(library_root: Path, guarantees: GuaranteeSet) -> list[GuaranteeItem]:
    """Console-script names and their entry-point targets, from the packaging metadata."""
    pyproject = library_root / "pyproject.toml"
    if not pyproject.is_file():
        return []
    data = tomllib.loads(pyproject.read_text(encoding="utf-8"))
    scripts: Mapping[str, str] = data.get("project", {}).get("scripts", {}) or {}
    items: list[GuaranteeItem] = []
    for name, target in sorted(scripts.items()):
        pattern = token_pattern(name)
        owners = tuple(row.row_id for row in guarantees.rows if pattern.search(row.doc_prose))
        if not owners:
            owners = (UNATTRIBUTED_ROW,)
        for row_id in owners:
            items.append(
                GuaranteeItem(
                    row_id=row_id,
                    kind="console_script",
                    value=name,
                    origin="pyproject:[project.scripts]",
                    detail={"target": target},
                )
            )
            items.append(
                GuaranteeItem(
                    row_id=row_id,
                    kind="object_reference",
                    value=target,
                    origin="pyproject:[project.scripts]",
                    detail={"console_script": name},
                )
            )
    return items


# --------------------------------------------------------------------------
# Corpora: explicit tracked-file listings and separately declared artifact roots
# --------------------------------------------------------------------------


@dataclass
class Corpus:
    name: str
    role: str
    root: Path
    restatement_prefixes: tuple[str, ...]
    restatement_reason: str | None
    source_files: list[str] = field(default_factory=list)
    artifact_files: list[str] = field(default_factory=list)
    snapshot_roots: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)
    _tracked: frozenset[str] | None = None

    def all_files(self) -> list[str]:
        """Tracked source first, then durable artifacts.

        Order matters downstream: the record aggregator caps how many files it
        keeps per item, and tracked source is the evidence a reader can open and
        review. It must never be crowded out by bulk run output.
        """
        tracked = sorted(set(self.source_files))
        artifacts = sorted(set(self.artifact_files) - set(tracked))
        return tracked + artifacts

    def tier(self, relpath: str) -> str:
        return "tracked" if relpath in self._tracked_index() else "artifact"

    def _tracked_index(self) -> frozenset[str]:
        if self._tracked is None:
            self._tracked = frozenset(self.source_files)
        return self._tracked

    def is_restatement(self, relpath: str) -> bool:
        return any(relpath.startswith(prefix) for prefix in self.restatement_prefixes)

    def describe(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "role": self.role,
            "root": str(self.root),
            "source_file_count": len(self.source_files),
            "artifact_file_count": len(self.artifact_files),
            "sealed_source_snapshot_roots": sorted(self.snapshot_roots),
            "sealed_source_snapshot_root_count": len(self.snapshot_roots),
            "restatement_prefixes": list(self.restatement_prefixes),
            "restatement_reason": self.restatement_reason,
            "notes": list(self.notes),
        }


def tracked_files(root: Path) -> list[str]:
    """Explicit NUL-delimited tracked-file listing. No globbing, no guessing."""
    env = dict(os.environ, GIT_OPTIONAL_LOCKS="0")
    result = subprocess.run(
        ["git", "--no-optional-locks", "ls-files", "-z"],
        cwd=root,
        env=env,
        capture_output=True,
        check=False,
    )
    if result.returncode != 0:
        raise ScanError(
            f"{root}: could not enumerate tracked files: {result.stderr.decode('utf-8', 'replace').strip()}"
        )
    return sorted(
        entry for entry in result.stdout.decode("utf-8", "surrogateescape").split("\0") if entry
    )


def detect_snapshot_roots(artifact_root: Path) -> list[Path]:
    """Find sealed source snapshots: nested checkout copies inside an artifact tree.

    A durable artifact tree does not contain a package build definition. A sealed
    copy of a repository always does. Directory naming conventions are a second,
    weaker signal, kept because some snapshots are captured without their build
    definition.
    """
    roots: list[Path] = []
    if not artifact_root.is_dir():
        return roots
    for dirpath, dirnames, filenames in os.walk(artifact_root):
        current = Path(dirpath)
        has_git_dir = ".git" in dirnames
        dirnames[:] = sorted(d for d in dirnames if d not in ALWAYS_SKIPPED_DIRS)
        if any(marker in filenames for marker in CHECKOUT_MARKERS) or has_git_dir:
            roots.append(current)
            dirnames[:] = []
            continue
        snapshot_children = [d for d in dirnames if d in SNAPSHOT_DIR_NAMES]
        for child in snapshot_children:
            roots.append(current / child)
            dirnames.remove(child)
    return roots


def walk_artifact_root(artifact_root: Path, snapshot_roots: Sequence[Path]) -> Iterator[Path]:
    sealed = tuple(str(p) + os.sep for p in snapshot_roots)
    sealed_exact = {str(p) for p in snapshot_roots}
    for dirpath, dirnames, filenames in os.walk(artifact_root):
        if dirpath in sealed_exact or dirpath.startswith(sealed):
            dirnames[:] = []
            continue
        dirnames[:] = sorted(d for d in dirnames if d not in ALWAYS_SKIPPED_DIRS)
        for name in sorted(filenames):
            yield Path(dirpath) / name


def build_corpus(spec: Mapping[str, Any], library_root: Path, scan_artifacts: bool) -> Corpus:
    raw_root = str(spec["root"])
    root = (library_root / raw_root[2:]).resolve() if raw_root.startswith("./") else Path(raw_root)
    if raw_root == ".":
        root = library_root.resolve()
    corpus = Corpus(
        name=spec["name"],
        role=spec["role"],
        root=root,
        restatement_prefixes=tuple(spec.get("restatement_paths") or ()),
        restatement_reason=spec.get("restatement_reason"),
    )
    if not root.is_dir():
        corpus.notes.append(f"root {root} does not exist; corpus contributes no evidence")
        return corpus

    source = spec.get("source") or {}
    mode = source.get("mode", "git-tracked")
    exclude_prefixes = tuple(source.get("exclude_prefixes") or ())
    include_suffixes = tuple(s.lower() for s in source.get("include_suffixes") or ())
    if mode == "git-tracked":
        listing = tracked_files(root)
    elif mode == "filesystem":
        listing = sorted(
            str(p.relative_to(root))
            for p in root.rglob("*")
            if p.is_file() and not any(part in ALWAYS_SKIPPED_DIRS for part in p.parts)
        )
    else:
        raise ScanError(f"{corpus.name}: unknown source mode {mode!r}")
    selected = [
        rel
        for rel in listing
        if not any(rel.startswith(prefix) for prefix in exclude_prefixes)
        and not any(part in ALWAYS_SKIPPED_DIRS for part in Path(rel).parts)
        and (not include_suffixes or Path(rel).suffix.lower() in include_suffixes)
    ]
    # A tracked path can be absent from the checkout. It is not evidence of
    # anything, and it is not a reason to abandon the corpus, but the count is
    # reported so an unexpectedly hollow tree is visible rather than assumed.
    corpus.source_files = [rel for rel in selected if (root / rel).is_file()]
    absent = len(selected) - len(corpus.source_files)
    if absent:
        corpus.notes.append(
            f"{absent} tracked paths are absent from the checkout and were not scanned"
        )

    declared_artifact_roots = list(spec.get("artifact_roots") or ())
    if not scan_artifacts and declared_artifact_roots:
        corpus.notes.append(
            "durable-artifact roots declared but not scanned (--no-artifacts): "
            + ", ".join(declared_artifact_roots)
        )
        return corpus
    for rel_root in declared_artifact_roots:
        artifact_root = root / rel_root
        if not artifact_root.is_dir():
            corpus.notes.append(f"declared artifact root {rel_root} is absent")
            continue
        snapshots = detect_snapshot_roots(artifact_root)
        corpus.snapshot_roots.extend(str(p.relative_to(root)) for p in snapshots)
        for path in walk_artifact_root(artifact_root, snapshots):
            corpus.artifact_files.append(str(path.relative_to(root)))
    corpus.artifact_files.sort()
    return corpus


# --------------------------------------------------------------------------
# Prefilter
# --------------------------------------------------------------------------


def build_needles(items: Sequence[GuaranteeItem]) -> list[str]:
    """Literal byte sequences worth looking for, longest first for stability."""
    needles: set[str] = set()
    for item in items:
        value = item.value
        if len(value) < 4:
            continue
        needles.add(value)
        if item.kind == "cli_invocation":
            needles.add(value.split()[0])
        if item.kind in {"object_reference"}:
            needles.add(value.replace(":", "."))
    return sorted(needles)


def prefilter(
    corpus: Corpus, needles: Sequence[str], workdir: Path, rg_binary: str | None
) -> tuple[list[str], dict[str, Any]]:
    """Narrow the corpus to files that contain at least one literal needle.

    The prefilter is a fixed-string multi-pattern search over raw bytes, which is
    the only stage that touches every byte of a large artifact tree. Precise
    structural analysis then runs on the survivors only.
    """
    files = corpus.all_files()
    stats: dict[str, Any] = {"input_file_count": len(files), "engine": "none"}
    if not files or not needles:
        stats["candidate_file_count"] = 0
        return [], stats

    if rg_binary is None:
        stats["engine"] = "python"
        candidates = [rel for rel in files if _python_prefilter(corpus.root / rel, needles)]
        stats["candidate_file_count"] = len(candidates)
        return candidates, stats

    pattern_file = workdir / f"needles.{corpus.name}.txt"
    pattern_file.write_text("\n".join(needles) + "\n", encoding="utf-8")

    base_command = [
        rg_binary,
        "--fixed-strings",
        "--files-with-matches",
        "--no-config",
        "--no-ignore",
        "--hidden",
        "--no-require-git",
        "--text",
        "--null",
        "--file",
        str(pattern_file),
    ]
    # Paths are passed as arguments in bounded batches rather than on stdin,
    # because ripgrep reads stdin as content to search, not as a path list.
    root_prefix = str(corpus.root) + os.sep
    candidates: set[str] = set()
    unreadable: list[str] = []
    batches = 0
    for batch in _batched(files, PREFILTER_BATCH_SIZE):
        batches += 1
        command = [*base_command, "--", *[str(corpus.root / rel) for rel in batch]]
        result = subprocess.run(command, capture_output=True, check=False)
        if result.returncode not in (0, 1):
            # ripgrep exits 2 when any single path could not be read. The batch's
            # other matches are still valid, so the unreadable path is recorded
            # and the scan continues rather than losing the whole corpus.
            unreadable.extend(
                line
                for line in result.stderr.decode("utf-8", "replace").splitlines()
                if line.strip()
            )
            if not result.stdout:
                continue
        for entry in result.stdout.decode("utf-8", "surrogateescape").split("\0"):
            entry = entry.strip("\n")
            if entry.startswith(root_prefix):
                candidates.add(entry[len(root_prefix) :])
    stats["engine"] = "ripgrep"
    stats["batches"] = batches
    stats["candidate_file_count"] = len(candidates)
    stats["unreadable_path_count"] = len(unreadable)
    stats["unreadable_paths"] = unreadable[:20]
    return sorted(candidates), stats


#: Paths per ripgrep invocation. Bounded so the argument vector stays well below
#: the platform limit for corpora with six-figure file counts.
PREFILTER_BATCH_SIZE = 1000


def _batched(values: Sequence[str], size: int) -> Iterator[list[str]]:
    for start in range(0, len(values), size):
        yield list(values[start : start + size])


def _python_prefilter(path: Path, needles: Sequence[str]) -> bool:
    encoded = [n.encode("utf-8") for n in needles]
    longest = max(len(n) for n in encoded)
    try:
        with path.open("rb") as handle:
            tail = b""
            while True:
                chunk = handle.read(1 << 20)
                if not chunk:
                    return False
                blob = tail + chunk
                if any(n in blob for n in encoded):
                    return True
                tail = blob[-(longest - 1) :] if longest > 1 else b""
    except OSError:
        return False


# --------------------------------------------------------------------------
# Loader attribution: where in this library does the dependency become live
# --------------------------------------------------------------------------


@dataclass
class LoaderIndex:
    """Dispatch and definition sites inside the library, found by parsing it."""

    literal_sites: dict[str, list[dict[str, Any]]] = field(default_factory=dict)
    definition_sites: dict[str, list[dict[str, Any]]] = field(default_factory=dict)
    environment_variables: dict[str, list[dict[str, Any]]] = field(default_factory=dict)
    parsed_files: int = 0
    parse_failures: list[str] = field(default_factory=list)

    def loader_for_literal(self, value: str) -> list[dict[str, Any]]:
        return self.literal_sites.get(value, [])

    def loader_for_symbol(self, value: str) -> list[dict[str, Any]]:
        return self.definition_sites.get(value, [])


def build_loader_index(library_root: Path, interesting: Sequence[str]) -> LoaderIndex:
    """Parse the library's own tracked Python and record where literals dispatch.

    The loader field of a record is evidence, not commentary: it points at the
    line in this library that reads the byte the downstream file wrote.
    """
    index = LoaderIndex()
    wanted = set(interesting)
    package_root = library_root / "feedbax"
    if not package_root.is_dir():
        index.parse_failures.append(f"{package_root} is absent; loader attribution unavailable")
        return index

    for path in sorted(package_root.rglob("*.py")):
        if any(part in ALWAYS_SKIPPED_DIRS for part in path.parts):
            continue
        try:
            tree = ast.parse(path.read_text(encoding="utf-8", errors="replace"), filename=str(path))
        except SyntaxError as exc:
            index.parse_failures.append(f"{path}: {exc}")
            continue
        index.parsed_files += 1
        rel = str(path.relative_to(library_root))

        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef | ast.FunctionDef | ast.AsyncFunctionDef):
                if node.name in wanted:
                    index.definition_sites.setdefault(node.name, []).append(
                        {"kind": "definition", "path": rel, "line": node.lineno}
                    )
            elif isinstance(node, ast.Assign):
                for target in node.targets:
                    if isinstance(target, ast.Name) and target.id in wanted:
                        index.definition_sites.setdefault(target.id, []).append(
                            {"kind": "definition", "path": rel, "line": target.lineno}
                        )
            elif isinstance(node, ast.Compare):
                operands = [node.left, *node.comparators]
                for operand in operands:
                    if isinstance(operand, ast.Constant) and isinstance(operand.value, str):
                        if operand.value in wanted:
                            index.literal_sites.setdefault(operand.value, []).append(
                                {"kind": "branch", "path": rel, "line": operand.lineno}
                            )
            elif isinstance(node, ast.Dict):
                for key in node.keys:
                    if isinstance(key, ast.Constant) and isinstance(key.value, str):
                        if key.value in wanted:
                            index.literal_sites.setdefault(key.value, []).append(
                                {"kind": "dispatch-table", "path": rel, "line": key.lineno}
                            )
            elif isinstance(node, ast.Subscript):
                base = node.value
                if isinstance(base, ast.Name) and base.id == "Literal":
                    for element in ast.walk(node.slice):
                        if isinstance(element, ast.Constant) and isinstance(element.value, str):
                            if element.value in wanted:
                                index.literal_sites.setdefault(element.value, []).append(
                                    {
                                        "kind": "type-discriminator",
                                        "path": rel,
                                        "line": element.lineno,
                                    }
                                )
            elif isinstance(node, ast.Constant) and isinstance(node.value, str):
                if ENV_VAR_RE.match(node.value):
                    index.environment_variables.setdefault(node.value, []).append(
                        {"kind": "environment-read", "path": rel, "line": node.lineno}
                    )

    for sites in index.literal_sites.values():
        sites.sort(key=lambda entry: (entry["path"], entry["line"], entry["kind"]))
        del sites[24:]
    for sites in index.definition_sites.values():
        sites.sort(key=lambda entry: (entry["path"], entry["line"]))
        del sites[8:]
    return index


# --------------------------------------------------------------------------
# Records
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class Record:
    corpus: str
    role: str
    row_id: str
    item: str
    item_kind: str
    channel: str
    path: str
    line: int | None
    json_pointer: str | None
    loader: tuple[str, ...]
    loader_kind: str
    evidence_class: str
    strength: str
    detail: str
    tier: str = "tracked"
    occurrences_in_file: int = 1

    def sort_key(self) -> tuple[Any, ...]:
        return (
            self.corpus,
            self.row_id,
            self.channel,
            self.path,
            self.line if self.line is not None else -1,
            self.json_pointer or "",
            self.item,
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "corpus": self.corpus,
            "role": self.role,
            "row_id": self.row_id,
            "item": self.item,
            "item_kind": self.item_kind,
            "channel": self.channel,
            "path": self.path,
            "line": self.line,
            "json_pointer": self.json_pointer,
            "loader": list(self.loader),
            "loader_kind": self.loader_kind,
            "evidence_class": self.evidence_class,
            "strength": self.strength,
            "detail": self.detail,
            "tier": self.tier,
            "occurrences_in_file": self.occurrences_in_file,
        }


#: Which channel carries a given kind of guaranteed item when it is found as a
#: string value inside a saved document.
DOCUMENT_CHANNEL_BY_KIND = {
    "public_name": "document-type-discriminator",
    "manifest_kind": "document-type-discriminator",
    "schema_id": "schema-id-string",
    "schema_family": "schema-family-prefix",
    "registry_identifier": "registry-identifier",
    "storage_directory": "registry-identifier",
    "case_id": "registry-identifier",
    "plugin_family_key": "plugin-family",
    "registry_method": "plugin-family",
    "console_script": "console-script",
    "cli_invocation": "console-script",
    "object_reference": "entry-point",
    "namespace": "dynamic-import-path",
    "field_name": "document-field-name",
    "environment_variable": "environment-variable",
}

#: Item kinds that are never matched against a saved document's string values or
#: object keys. A registry method is called on an object; it is not written into
#: a document, so matching one there is noise dressed as evidence.
DOCUMENT_EXCLUDED_KINDS = frozenset({"registry_method", "environment_variable"})

#: A registry identifier shorter than this is an ordinary English word
#: ("local", "all") long before it is evidence of a dependency.
MIN_REGISTRY_IDENTIFIER_LENGTH = 6

COMPOUND_NEEDLE_RE = re.compile(r"[._:\-]")


def is_unstructured_safe(needle: str) -> bool:
    """Is this needle specific enough to be believed in unstructured text?

    Structured channels match a whole value, so any needle is safe there. A line
    scan or a byte scan has no structure to lean on, so it may only look for
    needles that are unambiguous on sight: a compound identifier, an
    UPPER_SNAKE constant, or a long CamelCase name. A bare lowercase word like
    ``feedbax`` or ``resolve`` is a word, and matching it manufactures consumers.
    """
    if len(needle) < 8:
        return False
    if COMPOUND_NEEDLE_RE.search(needle):
        return True
    if needle.isupper():
        return True
    return needle[:1].isupper() and not needle.isupper()


def token_pattern(needle: str) -> re.Pattern[str]:
    """Match the needle only as a whole dotted/hyphenated token."""
    return re.compile(r"(?<![A-Za-z0-9_.\-])" + re.escape(needle) + r"(?![A-Za-z0-9_.\-])")


def document_value_kind_allowed(item: GuaranteeItem) -> bool:
    if item.kind in DOCUMENT_EXCLUDED_KINDS:
        return False
    if item.kind == "registry_identifier" and len(item.value) < MIN_REGISTRY_IDENTIFIER_LENGTH:
        return False
    if item.kind == "namespace" and "." not in item.value:
        return False
    return True


DISCRIMINATOR_KEYS = frozenset(
    {
        "kind",
        "type",
        "class",
        "__class__",
        "node_type",
        "component_type",
        "variant",
        "tag",
        "schema",
    }
)
PYTREE_KEYS = frozenset({"__qualname__", "qualname", "class_path", "type_path", "target", "cls"})
PYTREE_ROOT_KEYS = frozenset({"__module__", "module"})
PYTHON_LOADER = "python import machinery (import statement binds the name at call time)"
PYTREE_LOADER = (
    "pytree/checkpoint type fingerprints fold __module__ and __qualname__, so moving the class "
    "breaks the saved structure with no import and no literal name"
)
PICKLE_LOADER = "pickle.Unpickler.find_class resolves the stored (module, qualname) pair"


# --------------------------------------------------------------------------
# Analyzers
# --------------------------------------------------------------------------


@dataclass
class Analyzer:
    guarantees: GuaranteeSet
    items_by_value: dict[str, list[GuaranteeItem]]
    names_by_namespace: dict[str, list[GuaranteeItem]]
    loaders: LoaderIndex
    max_structural_bytes: int
    diagnostics: list[str] = field(default_factory=list)
    rg_binary: str | None = None
    pattern_file: Path | None = None
    _unstructured: dict[str, tuple[Any, list[GuaranteeItem]]] = field(default_factory=dict)
    _unstructured_ready: bool = False
    _combined: Any = None

    def unstructured_regex(self) -> Any:
        """One pass over a line instead of one pass per needle.

        Alternatives are ordered longest first so the longer identity wins:
        ``feedbax.spec.graph.v5`` must match as itself, not as its family prefix.
        """
        if self._combined is None:
            needles = sorted(self.unstructured_index(), key=lambda n: (-len(n), n))
            if not needles:
                self._combined = False
            else:
                self._combined = re.compile(
                    r"(?<![A-Za-z0-9_.\-])(?:"
                    + "|".join(re.escape(n) for n in needles)
                    + r")(?![A-Za-z0-9_.\-])"
                )
        return self._combined or None

    def unstructured_index(self) -> dict[str, tuple[Any, list[GuaranteeItem]]]:
        """Needles believable without structure, each with a whole-token matcher."""
        if not self._unstructured_ready:
            for needle, items in self.items_by_value.items():
                if not is_unstructured_safe(needle):
                    continue
                usable = [item for item in items if item.kind not in {"registry_method"}]
                if usable:
                    self._unstructured[needle] = (token_pattern(needle), usable)
            self._unstructured_ready = True
        return self._unstructured

    # -- dispatch --------------------------------------------------------

    def analyze(self, corpus: Corpus, relpath: str) -> list[Record]:
        path = corpus.root / relpath
        try:
            size = path.stat().st_size
        except OSError as exc:
            self.diagnostics.append(f"{corpus.name}:{relpath}: {exc}")
            return []
        suffix = path.suffix.lower()
        evidence = "restatement" if corpus.is_restatement(relpath) else "dependency"
        try:
            if suffix in PYTHON_SUFFIXES and size <= self.max_structural_bytes:
                return self._analyze_python(corpus, relpath, path, evidence)
            if suffix in TEXT_DOCUMENT_SUFFIXES and size <= self.max_structural_bytes:
                return self._analyze_document(corpus, relpath, path, suffix, evidence)
            if suffix in PICKLE_SUFFIXES:
                return self._analyze_pickle(corpus, relpath, path, size, evidence)
            if suffix in PLAIN_TEXT_SUFFIXES and size <= self.max_structural_bytes:
                return self._analyze_lines(corpus, relpath, path, evidence, "text")
            return self._analyze_bytes(corpus, relpath, path, evidence)
        except (OSError, UnicodeError) as exc:
            self.diagnostics.append(f"{corpus.name}:{relpath}: {type(exc).__name__}: {exc}")
            return []

    def _record(
        self,
        corpus: Corpus,
        relpath: str,
        item: GuaranteeItem,
        channel: str,
        *,
        line: int | None = None,
        pointer: str | None = None,
        evidence: str = "dependency",
        detail: str = "",
        loader: Sequence[str] | None = None,
        loader_kind: str = "",
    ) -> Record:
        if loader is None:
            loader, loader_kind = self._loader_for(item, channel)
        return Record(
            corpus=corpus.name,
            role=corpus.role,
            row_id=item.row_id,
            item=item.value,
            item_kind=item.kind,
            channel=channel,
            path=relpath,
            line=line,
            json_pointer=pointer,
            loader=tuple(loader),
            loader_kind=loader_kind,
            evidence_class=evidence,
            strength="namespace" if item.kind == "namespace" else "direct",
            detail=detail,
            tier=corpus.tier(relpath),
        )

    def _loader_for(self, item: GuaranteeItem, channel: str) -> tuple[list[str], str]:
        if channel in {"python-import", "python-attribute", "python-star-import"}:
            sites = self.loaders.loader_for_symbol(item.value)
            return (
                [f"{s['path']}:{s['line']}" for s in sites] or ["<definition site not found>"],
                PYTHON_LOADER,
            )
        if channel == "pickle-class-identity":
            sites = self.loaders.loader_for_symbol(item.value)
            return (
                [f"{s['path']}:{s['line']}" for s in sites] or ["<definition site not found>"],
                PICKLE_LOADER,
            )
        if channel == "pytree-type-fingerprint":
            sites = self.loaders.loader_for_symbol(item.value)
            return (
                [f"{s['path']}:{s['line']}" for s in sites] or ["<definition site not found>"],
                PYTREE_LOADER,
            )
        if channel == "environment-variable":
            sites = self.loaders.environment_variables.get(item.value, [])
            return (
                [f"{s['path']}:{s['line']}" for s in sites] or ["<read site not found>"],
                "os.environ read inside the library",
            )
        sites = self.loaders.loader_for_literal(item.value)
        if sites:
            kinds = sorted({s["kind"] for s in sites})
            return (
                [f"{s['path']}:{s['line']} ({s['kind']})" for s in sites],
                "library dispatch on the literal: " + ", ".join(kinds),
            )
        definition = self.loaders.loader_for_symbol(item.value)
        if definition:
            return (
                [f"{s['path']}:{s['line']}" for s in definition],
                "library definition; no literal dispatch site found",
            )
        return (["<no dispatch site found in library source>"], "unattributed")

    # -- python ----------------------------------------------------------

    def _analyze_python(
        self, corpus: Corpus, relpath: str, path: Path, evidence: str
    ) -> list[Record]:
        text = path.read_text(encoding="utf-8", errors="replace")
        try:
            tree = ast.parse(text, filename=relpath)
        except SyntaxError as exc:
            self.diagnostics.append(f"{corpus.name}:{relpath}: syntax error, line scan used: {exc}")
            return self._analyze_lines(corpus, relpath, path, evidence, "python-unparsed")

        records: list[Record] = []
        module_aliases: dict[str, str] = {}
        imported_symbols: dict[str, GuaranteeItem] = {}
        docstring_nodes: set[int] = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Module | ast.ClassDef | ast.FunctionDef | ast.AsyncFunctionDef):
                body = getattr(node, "body", None)
                if (
                    body
                    and isinstance(body[0], ast.Expr)
                    and isinstance(body[0].value, ast.Constant)
                ):
                    if isinstance(body[0].value.value, str):
                        docstring_nodes.add(id(body[0].value))

        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    if not _is_governed_module(alias.name):
                        continue
                    # Without an alias, `import a.b.c` binds only `a`, and the
                    # attribute chain spells out the rest; with one, the alias
                    # stands for the whole dotted path.
                    if alias.asname:
                        module_aliases[alias.asname] = alias.name
                    else:
                        head = alias.name.split(".")[0]
                        module_aliases[head] = head
                    for item in self.items_by_value.get(alias.name, ()):
                        if item.kind == "namespace":
                            records.append(
                                self._record(
                                    corpus,
                                    relpath,
                                    item,
                                    "python-import",
                                    line=node.lineno,
                                    evidence=evidence,
                                    detail=f"import {alias.name}"
                                    + (f" as {alias.asname}" if alias.asname else ""),
                                )
                            )
            elif isinstance(node, ast.ImportFrom):
                module = node.module or ""
                if node.level or not _is_governed_module(module):
                    continue
                for alias in node.names:
                    if alias.name == "*":
                        for item in self.names_by_namespace.get(module, ()):
                            records.append(
                                self._record(
                                    corpus,
                                    relpath,
                                    item,
                                    "python-star-import",
                                    line=node.lineno,
                                    evidence=evidence,
                                    detail=f"from {module} import * (name reachable, not proven used)",
                                )
                            )
                        continue
                    for item in self.items_by_value.get(alias.name, ()):
                        if item.kind not in {"public_name", "namespace"}:
                            continue
                        if item.kind == "public_name":
                            imported_symbols.setdefault(alias.asname or alias.name, item)
                        declared = module in (item.detail.get("namespaces") or [])
                        records.append(
                            self._record(
                                corpus,
                                relpath,
                                item,
                                "python-import",
                                line=node.lineno,
                                evidence=evidence,
                                detail=(
                                    f"from {module} import {alias.name}"
                                    + (f" as {alias.asname}" if alias.asname else "")
                                    + (
                                        ""
                                        if declared
                                        else "  [imported from a re-export namespace,"
                                        " not the declared sub-namespace]"
                                    )
                                ),
                            )
                        )
                    submodule = f"{module}.{alias.name}"
                    for item in self.items_by_value.get(submodule, ()):
                        if item.kind == "namespace":
                            records.append(
                                self._record(
                                    corpus,
                                    relpath,
                                    item,
                                    "python-import",
                                    line=node.lineno,
                                    evidence=evidence,
                                    detail=f"from {module} import {alias.name} (submodule)",
                                )
                            )

        for node in ast.walk(tree):
            if isinstance(node, ast.Attribute):
                base, attrs = _flatten_attribute(node)
                if base is None or not attrs:
                    continue
                bound = imported_symbols.get(base)
                if bound is not None and len(attrs) == 1:
                    for item in self.items_by_value.get(attrs[0], ()):
                        if item.kind != "registry_method":
                            continue
                        records.append(
                            self._record(
                                corpus,
                                relpath,
                                item,
                                "plugin-family",
                                line=node.lineno,
                                evidence=evidence,
                                detail=f"guaranteed registry method {base}.{attrs[0]}() "
                                f"called on imported {bound.value}",
                            )
                        )
                target = module_aliases.get(base)
                if target is None:
                    continue
                symbol = attrs[-1]
                resolved = ".".join([target, *attrs])
                for item in self.items_by_value.get(symbol, ()):
                    if item.kind != "public_name":
                        continue
                    records.append(
                        self._record(
                            corpus,
                            relpath,
                            item,
                            "python-attribute",
                            line=node.lineno,
                            evidence=evidence,
                            detail=f"{base}.{'.'.join(attrs)} resolves to {resolved}",
                        )
                    )
            elif isinstance(node, ast.Constant) and isinstance(node.value, str):
                in_docstring = id(node) in docstring_nodes
                records.extend(
                    self._records_for_string(
                        corpus,
                        relpath,
                        None,
                        node.value,
                        line=node.lineno,
                        pointer=None,
                        evidence="prose" if in_docstring else evidence,
                        origin="python string literal",
                    )
                )
        return records

    # -- structured documents -------------------------------------------

    def _analyze_document(
        self, corpus: Corpus, relpath: str, path: Path, suffix: str, evidence: str
    ) -> list[Record]:
        raw = path.read_text(encoding="utf-8", errors="replace")
        documents: list[tuple[str, Any]] = []
        try:
            if suffix == ".json":
                documents.append(("", json.loads(raw)))
            elif suffix == ".jsonl":
                for offset, line in enumerate(raw.splitlines(), start=1):
                    if line.strip():
                        documents.append((f"#line={offset}", json.loads(line)))
            elif suffix == ".toml":
                documents.append(("", tomllib.loads(raw)))
            else:
                import yaml  # noqa: PLC0415 - optional dependency, reported when absent

                for offset, doc in enumerate(yaml.safe_load_all(raw)):
                    documents.append((f"#doc={offset}", doc))
        except Exception as exc:  # noqa: BLE001 - parse failure is evidence, not a crash
            self.diagnostics.append(
                f"{corpus.name}:{relpath}: structural parse failed ({type(exc).__name__}); "
                "line scan used"
            )
            return self._analyze_lines(corpus, relpath, path, evidence, "document-unparsed")

        records: list[Record] = []
        for prefix, document in documents:
            records.extend(self._walk_document(corpus, relpath, document, prefix, evidence))
        return records

    def _walk_document(
        self, corpus: Corpus, relpath: str, node: Any, pointer: str, evidence: str
    ) -> list[Record]:
        records: list[Record] = []
        stack: list[tuple[str, Any, str | None]] = [(pointer, node, None)]
        while stack:
            current_pointer, current, parent_key = stack.pop()
            if isinstance(current, dict):
                for key in sorted(current, key=str):
                    child_pointer = f"{current_pointer}/{_escape_pointer(str(key))}"
                    for item in self.items_by_value.get(str(key), ()):
                        if item.kind in {"field_name", "plugin_family_key"}:
                            records.append(
                                self._record(
                                    corpus,
                                    relpath,
                                    item,
                                    "document-field-name"
                                    if item.kind == "field_name"
                                    else "plugin-family",
                                    pointer=child_pointer,
                                    evidence=evidence,
                                    detail=f"object key {key!r}",
                                )
                            )
                    stack.append((child_pointer, current[key], str(key)))
            elif isinstance(current, list):
                for index in range(len(current) - 1, -1, -1):
                    stack.append((f"{current_pointer}/{index}", current[index], parent_key))
            elif isinstance(current, str):
                records.extend(
                    self._records_for_string(
                        corpus,
                        relpath,
                        parent_key,
                        current,
                        line=None,
                        pointer=current_pointer,
                        evidence=evidence,
                        origin="document string value",
                    )
                )
        return records

    # -- shared string classification -----------------------------------

    def _records_for_string(
        self,
        corpus: Corpus,
        relpath: str,
        key: str | None,
        value: str,
        *,
        line: int | None,
        pointer: str | None,
        evidence: str,
        origin: str,
    ) -> list[Record]:
        records: list[Record] = []
        stripped = value.strip()
        structured = bool(stripped) and not any(ch.isspace() for ch in stripped)
        prose_key = key is not None and key.lower() in PROSE_KEYS

        if structured:
            exact = [
                item
                for item in self.items_by_value.get(stripped, ())
                if document_value_kind_allowed(item)
            ]
            for item in exact:
                channel = DOCUMENT_CHANNEL_BY_KIND.get(item.kind)
                if channel is None:
                    continue
                if item.kind == "public_name":
                    channel = "document-type-discriminator"
                effective = "prose" if prose_key else evidence
                records.append(
                    self._record(
                        corpus,
                        relpath,
                        item,
                        channel,
                        line=line,
                        pointer=pointer,
                        evidence=effective,
                        detail=f"{origin}; exact identity"
                        + (f"; key={key!r}" if key else "")
                        + ("; discriminator key" if key in DISCRIMINATOR_KEYS else ""),
                    )
                )
            if not exact:
                family = SCHEMA_VERSION_TAIL_RE.sub("", stripped)
                if family != stripped:
                    for item in self.items_by_value.get(family, ()):
                        if item.kind != "schema_family":
                            continue
                        records.append(
                            self._record(
                                corpus,
                                relpath,
                                item,
                                "schema-family-prefix",
                                line=line,
                                pointer=pointer,
                                evidence="prose" if prose_key else evidence,
                                detail=f"{origin}; live version {stripped!r} of guaranteed family",
                            )
                        )
                match = OBJECT_REF_RE.match(stripped)
                if match and _is_governed_module(match.group("module")):
                    attr = match.group("attr")
                    separator = ":" if ":" in stripped else "."
                    for item in self.items_by_value.get(attr, ()):
                        if item.kind != "public_name":
                            continue
                        channel = (
                            "entry-point"
                            if separator == ":"
                            else (
                                "pytree-type-fingerprint"
                                if (key or "") in PYTREE_KEYS or attr[:1].isupper()
                                else "dynamic-import-path"
                            )
                        )
                        records.append(
                            self._record(
                                corpus,
                                relpath,
                                item,
                                channel,
                                line=line,
                                pointer=pointer,
                                evidence="prose" if prose_key else evidence,
                                detail=f"{origin}; stored object path {stripped!r}"
                                + (f"; key={key!r}" if key else ""),
                            )
                        )
                    if not self.items_by_value.get(attr):
                        for item in self.items_by_value.get(match.group("module"), ()):
                            if item.kind != "namespace" or not document_value_kind_allowed(item):
                                continue
                            records.append(
                                self._record(
                                    corpus,
                                    relpath,
                                    item,
                                    "dynamic-import-path",
                                    line=line,
                                    pointer=pointer,
                                    evidence="prose" if prose_key else evidence,
                                    detail=f"{origin}; stored module path {stripped!r}",
                                )
                            )
            return records

        # Free text: a guaranteed name inside a sentence is prose, never a
        # dependency. It is still recorded, so the distinction is auditable.
        combined = self.unstructured_regex()
        index = self.unstructured_index()
        # One record per identity per position, not one per repetition: a long
        # note that names the same thing five times is still one mention.
        seen: set[tuple[str, str]] = set()
        for match in combined.finditer(value) if combined else ():
            for item in index[match.group(0)][1]:
                channel = DOCUMENT_CHANNEL_BY_KIND.get(item.kind)
                if channel is None or (item.row_id, item.value) in seen:
                    continue
                seen.add((item.row_id, item.value))
                records.append(
                    self._record(
                        corpus,
                        relpath,
                        item,
                        channel,
                        line=line,
                        pointer=pointer,
                        evidence="prose",
                        detail=f"{origin}; name appears inside free text"
                        + (f"; key={key!r}" if key else ""),
                    )
                )
        return records

    # -- pickle ----------------------------------------------------------

    def _analyze_pickle(
        self, corpus: Corpus, relpath: str, path: Path, size: int, evidence: str
    ) -> list[Record]:
        if size > self.max_structural_bytes:
            return self._analyze_bytes(corpus, relpath, path, evidence)
        data = path.read_bytes()
        pairs: list[tuple[str, str]] = []
        try:
            recent: list[str] = []
            for opcode, argument, _pos in pickletools.genops(io.BytesIO(data)):
                if opcode.name in {"GLOBAL", "STACK_GLOBAL"}:
                    if opcode.name == "GLOBAL" and isinstance(argument, str):
                        module, _, qualname = argument.partition(" ")
                        pairs.append((module, qualname))
                    elif len(recent) >= 2:
                        pairs.append((recent[-2], recent[-1]))
                elif isinstance(argument, str):
                    recent.append(argument)
                    del recent[:-4]
        except Exception as exc:  # noqa: BLE001 - not a pickle, or truncated
            self.diagnostics.append(
                f"{corpus.name}:{relpath}: pickle walk failed ({type(exc).__name__}); byte scan used"
            )
            return self._analyze_bytes(corpus, relpath, path, evidence)

        records: list[Record] = []
        seen: set[tuple[str, str]] = set()
        for module, qualname in pairs:
            if not _is_governed_module(module) or (module, qualname) in seen:
                continue
            seen.add((module, qualname))
            emitted = False
            for item in self.items_by_value.get(qualname, ()):
                if item.kind != "public_name":
                    continue
                emitted = True
                records.append(
                    self._record(
                        corpus,
                        relpath,
                        item,
                        "pickle-class-identity",
                        evidence=evidence,
                        detail=f"pickled class identity {module}.{qualname}",
                    )
                )
            if not emitted:
                for item in self.items_by_value.get(module, ()):
                    if item.kind != "namespace":
                        continue
                    records.append(
                        self._record(
                            corpus,
                            relpath,
                            item,
                            "pickle-class-identity",
                            evidence=evidence,
                            detail=f"pickled class identity {module}.{qualname} "
                            "(class is not itself a guaranteed name; the module is)",
                        )
                    )
        return records

    # -- fallbacks -------------------------------------------------------

    def _analyze_lines(
        self, corpus: Corpus, relpath: str, path: Path, evidence: str, mode: str
    ) -> list[Record]:
        records: list[Record] = []
        combined = self.unstructured_regex()
        if combined is None:
            return records
        index = self.unstructured_index()
        seen: set[tuple[str, str, int]] = set()
        with path.open("r", encoding="utf-8", errors="replace") as handle:
            for number, line in enumerate(handle, start=1):
                for match in combined.finditer(line):
                    for item in index[match.group(0)][1]:
                        channel = DOCUMENT_CHANNEL_BY_KIND.get(item.kind)
                        if channel is None or (item.row_id, item.value, number) in seen:
                            continue
                        seen.add((item.row_id, item.value, number))
                        records.append(
                            self._record(
                                corpus,
                                relpath,
                                item,
                                channel,
                                line=number,
                                evidence=evidence,
                                detail=f"line scan ({mode}); no structural position available",
                            )
                        )
        return records

    def _analyze_bytes(
        self, corpus: Corpus, relpath: str, path: Path, evidence: str
    ) -> list[Record]:
        """Locate guaranteed identities inside an opaque or oversized store.

        There is no structure to walk here, so the record carries a byte offset
        instead of a line or a JSON pointer. Matching is delegated to the same
        fixed-string engine as the prefilter, because a per-needle Python scan
        over multi-gigabyte artifacts is quadratic in the needle count.
        """
        index = self.unstructured_index()
        if not index:
            return []
        hits: dict[str, int] = {}
        if self.rg_binary and self.pattern_file is not None:
            command = [
                self.rg_binary,
                "--fixed-strings",
                "--only-matching",
                "--byte-offset",
                "--no-filename",
                "--no-config",
                "--no-messages",
                "--no-ignore",
                "--text",
                "--max-count",
                str(BYTE_SCAN_MAX_MATCHES),
                "--file",
                str(self.pattern_file),
                "--",
                str(path),
            ]
            result = subprocess.run(command, capture_output=True, check=False)
            if result.returncode not in (0, 1):
                self.diagnostics.append(
                    f"{corpus.name}:{relpath}: byte scan failed rc={result.returncode}"
                )
                return []
            for entry in result.stdout.decode("utf-8", "replace").splitlines():
                offset, _, needle = entry.partition(":")
                if needle in index and needle not in hits:
                    try:
                        hits[needle] = int(offset)
                    except ValueError:
                        hits[needle] = -1
        else:
            hits = self._python_byte_scan(path, index)

        records: list[Record] = []
        for needle in sorted(hits):
            for item in index[needle][1]:
                channel = DOCUMENT_CHANNEL_BY_KIND.get(item.kind)
                if channel is None:
                    continue
                records.append(
                    self._record(
                        corpus,
                        relpath,
                        item,
                        "binary-store-identity",
                        evidence=evidence,
                        detail=f"byte offset {hits[needle]} in an opaque or oversized store; "
                        f"nominal channel {channel}",
                    )
                )
        return records

    def _python_byte_scan(self, path: Path, index: Mapping[str, Any]) -> dict[str, int]:
        encoded = {needle: needle.encode("utf-8") for needle in index}
        longest = max(len(value) for value in encoded.values())
        found: dict[str, int] = {}
        with path.open("rb") as handle:
            base = 0
            tail = b""
            while True:
                chunk = handle.read(4 << 20)
                if not chunk:
                    break
                blob = tail + chunk
                for needle, raw in encoded.items():
                    if needle in found:
                        continue
                    position = blob.find(raw)
                    if position >= 0:
                        found[needle] = base - len(tail) + position
                base += len(chunk)
                tail = blob[-(longest - 1) :] if longest > 1 else b""
        return found


#: Byte-scan matches read from one oversized or opaque file. The scan only needs
#: to establish which identities are present and where one of them sits, so it is
#: bounded rather than exhaustive.
BYTE_SCAN_MAX_MATCHES = 20000


def _is_governed_module(name: str) -> bool:
    return any(name == root or name.startswith(root + ".") for root in GOVERNED_ROOT_PACKAGES)


def _flatten_attribute(node: ast.Attribute) -> tuple[str | None, list[str]]:
    attrs: list[str] = []
    current: ast.expr = node
    while isinstance(current, ast.Attribute):
        attrs.append(current.attr)
        current = current.value
    if isinstance(current, ast.Name):
        return current.id, list(reversed(attrs))
    return None, []


def _escape_pointer(token: str) -> str:
    return token.replace("~", "~0").replace("/", "~1")


# --------------------------------------------------------------------------
# Controls and verdicts
# --------------------------------------------------------------------------

#: Distinct files kept per guaranteed item per channel. A dependency is a thing
#: consumed in a place; the ten-thousandth file that consumes the same item
#: through the same channel adds a count, not an argument. The overflow is
#: counted exactly, so nothing is silently lost.
MAX_FILES_PER_ITEM_CHANNEL = 100


class RecordAggregator:
    """Collapse repeated evidence to one record per file, with exact counts.

    Raw occurrences run to millions across a multi-gigabyte artifact tree. Kept
    verbatim they exhaust memory and produce an output nobody can read, so
    identical evidence from one file collapses to one record carrying its own
    occurrence count, and the per-item file list is bounded with an exact
    overflow count rather than truncated silently.
    """

    def __init__(self, limit: int = MAX_FILES_PER_ITEM_CHANNEL) -> None:
        self._limit = limit
        self._groups: dict[tuple[str, ...], dict[str, Any]] = {}

    def add(self, record: Record) -> None:
        key = (
            record.corpus,
            record.row_id,
            record.item,
            record.item_kind,
            record.channel,
            record.evidence_class,
            record.tier,
        )
        group = self._groups.get(key)
        if group is None:
            group = {"files": {}, "omitted_files": 0, "omitted_occurrences": 0}
            self._groups[key] = group
        files: dict[str, Record] = group["files"]
        existing = files.get(record.path)
        if existing is not None:
            files[record.path] = replace_record(
                existing, occurrences_in_file=existing.occurrences_in_file + 1
            )
            return
        if len(files) >= self._limit:
            group["omitted_files"] += 1
            group["omitted_occurrences"] += 1
            return
        files[record.path] = record

    def extend(self, records: Iterable[Record]) -> None:
        for record in records:
            self.add(record)

    def records(self) -> list[Record]:
        out: list[Record] = []
        for group in self._groups.values():
            out.extend(group["files"].values())
        out.sort(key=Record.sort_key)
        return out

    def truncation(self) -> dict[str, Any]:
        omitted = [
            {
                "corpus": key[0],
                "row_id": key[1],
                "item": key[2],
                "channel": key[4],
                "evidence_class": key[5],
                "tier": key[6],
                "kept_files": len(group["files"]),
                "omitted_files": group["omitted_files"],
            }
            for key, group in sorted(self._groups.items())
            if group["omitted_files"]
        ]
        return {
            "files_per_item_channel_limit": self._limit,
            "note": (
                "The limit applies per item, per channel, and per tier, so bulk run output "
                "never crowds out tracked source. Evidence beyond it is counted, not "
                "discarded: omitted_files is the exact number of further distinct files "
                "carrying that item on that channel in that tier."
            ),
            "groups_truncated": len(omitted),
            "total_omitted_files": sum(entry["omitted_files"] for entry in omitted),
            "detail": omitted[:200],
        }

    def file_counts(self, corpus: str, row_id: str, evidence_class: str) -> dict[str, int]:
        """Distinct files actually kept, and the overflow, kept apart.

        The overflow cannot be deduplicated across items and channels, because
        the paths behind it were never retained. Adding the two would state a
        distinct-file count that is not one, so they are reported separately.
        """
        paths: set[str] = set()
        omitted = 0
        for key, group in self._groups.items():
            if key[0] != corpus or key[1] != row_id or key[5] != evidence_class:
                continue
            paths.update(group["files"])
            omitted += group["omitted_files"]
        return {"kept_distinct_files": len(paths), "omitted_file_hits": omitted}


def replace_record(record: Record, **changes: Any) -> Record:
    values = {
        "corpus": record.corpus,
        "role": record.role,
        "row_id": record.row_id,
        "item": record.item,
        "item_kind": record.item_kind,
        "channel": record.channel,
        "path": record.path,
        "line": record.line,
        "json_pointer": record.json_pointer,
        "loader": record.loader,
        "loader_kind": record.loader_kind,
        "evidence_class": record.evidence_class,
        "strength": record.strength,
        "detail": record.detail,
        "tier": record.tier,
        "occurrences_in_file": record.occurrences_in_file,
    }
    values.update(changes)
    return Record(**values)


#: What each channel's positive control is, so a zero from that channel can be
#: told apart from a channel that never worked. A channel that returns zero on
#: its own control has its zeros elsewhere withheld rather than reported.
CONTROL_DESCRIPTIONS = {
    "python-import": "from feedbax.contracts.graph import GraphSpec",
    "python-attribute": "import feedbax.contracts.graph as graph_mod; graph_mod.GraphSpec",
    "python-star-import": "from feedbax.lowering import *",
    "schema-id-string": "an exact guaranteed schema identity as a JSON string value",
    "schema-family-prefix": "a live version of a guaranteed schema family that the manifest does not enumerate",
    "document-type-discriminator": '{"kind": "GraphSpec"} nested under an arbitrary key',
    "document-field-name": "a nested field name that implies a spec type without naming it",
    "registry-identifier": "a component type id resolved dynamically from the component registry",
    "plugin-family": "a plugin registry family key",
    "entry-point": "a console-script target of the form module:object",
    "dynamic-import-path": "a stored module path handed to dynamic import",
    "pickle-class-identity": "a pickle GLOBAL opcode naming a library class",
    "pytree-type-fingerprint": "__module__/__qualname__ pair in a checkpoint structure record",
    "console-script": "a console-script name in a shell command",
    "environment-variable": "a FEEDBAX_* environment variable read by the library",
    "binary-store-identity": "a guaranteed identity inside an opaque binary store",
}


def evaluate_controls(control_records: Sequence[Record]) -> list[dict[str, Any]]:
    counts: dict[str, int] = dict.fromkeys(CHANNELS, 0)
    for record in control_records:
        if record.evidence_class == "dependency":
            counts[record.channel] = counts.get(record.channel, 0) + 1
    results = []
    for channel in CHANNELS:
        observed = counts.get(channel, 0)
        results.append(
            {
                "channel": channel,
                "control": CONTROL_DESCRIPTIONS.get(channel, "<undeclared>"),
                "observed_records": observed,
                "status": "pass" if observed > 0 else "broken",
            }
        )
    return results


def build_verdicts(
    guarantees: GuaranteeSet,
    records: Sequence[Record],
    corpora: Sequence[Corpus],
    healthy_channels: set[str],
    aggregator: RecordAggregator | None = None,
) -> dict[str, Any]:
    verdicts: dict[str, Any] = {}
    for corpus in corpora:
        per_row: dict[str, Any] = {}
        for row in guarantees.rows:
            live = [
                record
                for record in records
                if record.corpus == corpus.name
                and record.row_id == row.row_id
                and record.evidence_class == "dependency"
            ]
            hits = [record for record in live if record.strength == "direct"]
            namespace_only = [record for record in live if record.strength == "namespace"]
            channels = sorted({record.channel for record in hits})
            other = {
                klass: sum(
                    1
                    for record in records
                    if record.corpus == corpus.name
                    and record.row_id == row.row_id
                    and record.evidence_class == klass
                )
                for klass in ("prose", "restatement")
            }
            unhealthy = sorted(set(CHANNELS) - healthy_channels)
            if hits:
                verdict = "consumed"
            elif unhealthy:
                verdict = "indeterminate"
            elif namespace_only:
                verdict = "namespace-reference-only"
            else:
                verdict = "unconsumed"
            per_row[row.row_id] = {
                "verdict": verdict,
                "record_count": len(hits),
                "channels": channels,
                "items": sorted({record.item for record in hits})[:40],
                "distinct_item_count": len({record.item for record in hits}),
                "files": sorted({record.path for record in hits})[:20],
                **(
                    aggregator.file_counts(corpus.name, row.row_id, "dependency")
                    if aggregator is not None
                    else {
                        "kept_distinct_files": len({record.path for record in hits}),
                        "omitted_file_hits": 0,
                    }
                ),
                "occurrence_count": sum(record.occurrences_in_file for record in hits),
                "namespace_reference_records": len(namespace_only),
                "namespace_reference_files": sorted({r.path for r in namespace_only})[:10],
                "excluded_evidence": other,
                "withheld_because_channels_broken": unhealthy if not hits else [],
            }
        verdicts[corpus.name] = {"role": corpus.role, "rows": per_row}
    return verdicts


# --------------------------------------------------------------------------
# Entry point
# --------------------------------------------------------------------------


def prepare_items_and_loaders(
    library_root: Path,
    guarantees: GuaranteeSet,
    runtime: Mapping[str, Any],
    *,
    include_ungoverned: bool = False,
) -> tuple[list[GuaranteeItem], LoaderIndex]:
    """Assemble everything worth searching for, and where each of it dispatches.

    The loader index is built from the assembled items, and the environment
    variables it discovers become items in turn, so the two are produced together
    rather than by a caller that has to remember the order.
    """
    items = list(guarantees.items)
    items.extend(runtime_items(runtime, guarantees, include_ungoverned=include_ungoverned))
    items.extend(console_script_items(library_root, guarantees))

    loaders = build_loader_index(
        library_root, [item.value for item in items] + [row.row_id for row in guarantees.rows]
    )
    for name in sorted(loaders.environment_variables):
        owners = [row.row_id for row in guarantees.rows if name in row.doc_prose] or [
            UNATTRIBUTED_ROW
        ]
        for row_id in owners:
            items.append(
                GuaranteeItem(
                    row_id=row_id,
                    kind="environment_variable",
                    value=name,
                    origin="library-source:FEEDBAX_* literal",
                )
            )
    return sorted(_dedupe_items(items), key=GuaranteeItem.sort_key), loaders


def index_items_by_value(items: Sequence[GuaranteeItem]) -> dict[str, list[GuaranteeItem]]:
    index: dict[str, list[GuaranteeItem]] = {}
    for item in items:
        index.setdefault(item.value, []).append(item)
    return index


def index_names_by_namespace(items: Sequence[GuaranteeItem]) -> dict[str, list[GuaranteeItem]]:
    index: dict[str, list[GuaranteeItem]] = {}
    for item in items:
        if item.kind != "public_name":
            continue
        for namespace in item.detail.get("namespaces", ()) or ():
            index.setdefault(namespace, []).append(item)
    return index


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Record what actually depends on each guaranteed interface.",
    )
    parser.add_argument("--library-root", default=None, help="Feedbax checkout root")
    parser.add_argument("--manifest", default=None, help="policy manifest path")
    parser.add_argument("--policy-doc", default=None, help="policy document path")
    parser.add_argument("--corpora", default=None, help="scan-target configuration")
    parser.add_argument("--out", default=None, help="output file")
    parser.add_argument(
        "--no-artifacts",
        action="store_true",
        help="skip declared durable-artifact roots (source roots only)",
    )
    parser.add_argument(
        "--only-corpus",
        action="append",
        default=None,
        help="restrict to named corpora (repeatable); the control corpus always runs",
    )
    parser.add_argument(
        "--include-ungoverned",
        action="store_true",
        help="also search live library identities that belong to no guaranteed row",
    )
    parser.add_argument(
        "--max-structural-bytes",
        type=int,
        default=64 * 1024 * 1024,
        help="largest file given a structural parse; larger files get a byte scan",
    )
    parser.add_argument("--workdir", default=None, help="scratch directory for prefilter inputs")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    library_root = (
        Path(args.library_root).resolve()
        if args.library_root
        else Path(__file__).resolve().parent.parent
    )
    manifest_path = Path(args.manifest) if args.manifest else library_root / DEFAULT_MANIFEST
    doc_path = Path(args.policy_doc) if args.policy_doc else library_root / DEFAULT_POLICY_DOC
    corpora_path = Path(args.corpora) if args.corpora else library_root / DEFAULT_CORPORA
    out_path = Path(args.out) if args.out else library_root / DEFAULT_OUTPUT

    guarantees = GuaranteeSet.load(manifest_path, doc_path)
    runtime = collect_runtime_facts(library_root)

    items, loaders = prepare_items_and_loaders(
        library_root, guarantees, runtime, include_ungoverned=args.include_ungoverned
    )
    items_by_value = index_items_by_value(items)
    names_by_namespace = index_names_by_namespace(items)

    config = json.loads(corpora_path.read_text(encoding="utf-8"))
    selected = set(args.only_corpus) if args.only_corpus else None
    corpus_specs = [
        spec for spec in config["corpora"] if selected is None or spec["name"] in selected
    ]
    control_spec = config["control_corpus"]

    workdir = Path(args.workdir) if args.workdir else out_path.parent / ".guarantee_scan_work"
    workdir.mkdir(parents=True, exist_ok=True)

    rg_binary = shutil.which("rg")
    analyzer = Analyzer(
        guarantees=guarantees,
        items_by_value=items_by_value,
        names_by_namespace=names_by_namespace,
        loaders=loaders,
        max_structural_bytes=args.max_structural_bytes,
        rg_binary=rg_binary,
    )
    needles = build_needles(items)
    # The byte scan uses only the needles that are believable without structure,
    # so an opaque store is never credited with a match on a bare word.
    unstructured_pattern_file = workdir / "needles.unstructured.txt"
    unstructured_pattern_file.write_text(
        "\n".join(sorted(analyzer.unstructured_index())) + "\n", encoding="utf-8"
    )
    analyzer.pattern_file = unstructured_pattern_file

    corpora: list[Corpus] = []
    aggregator = RecordAggregator()
    prefilter_stats: dict[str, Any] = {}

    control_corpus = build_corpus(control_spec, library_root, scan_artifacts=True)
    control_records: list[Record] = []
    for rel in control_corpus.all_files():
        control_records.extend(analyzer.analyze(control_corpus, rel))
    controls = evaluate_controls(control_records)
    healthy = {entry["channel"] for entry in controls if entry["status"] == "pass"}

    declarations = (manifest_path.resolve(), doc_path.resolve())
    for spec in corpus_specs:
        corpus = build_corpus(spec, library_root, scan_artifacts=not args.no_artifacts)
        # The files the guarantee set was read from declare the guarantee. Finding
        # every guaranteed name in them proves only that they are the source.
        corpus.restatement_prefixes += tuple(
            str(path.relative_to(corpus.root))
            for path in declarations
            if path.is_relative_to(corpus.root)
        )
        candidates, stats = prefilter(corpus, needles, workdir, rg_binary)
        prefilter_stats[corpus.name] = stats
        for rel in candidates:
            aggregator.extend(analyzer.analyze(corpus, rel))
        corpora.append(corpus)

    records = aggregator.records()
    control_records.sort(key=Record.sort_key)

    payload = {
        "schema_id": OUTPUT_SCHEMA_ID,
        "schema_version": OUTPUT_SCHEMA_VERSION,
        "inputs": {
            "library_root": str(library_root),
            "policy_manifest": {
                "path": guarantees.manifest_path,
                "sha256": guarantees.manifest_sha256,
            },
            "policy_document": {"path": guarantees.doc_path, "sha256": guarantees.doc_sha256},
            "corpora_config": str(corpora_path),
            "prefilter_engine": "ripgrep" if rg_binary else "python",
            "max_structural_bytes": args.max_structural_bytes,
            "artifact_roots_scanned": not args.no_artifacts,
            "ungoverned_identities_searched": args.include_ungoverned,
        },
        "guarantee_set": {
            "crosscheck": guarantees.crosscheck,
            "rows": [
                {
                    "row_id": row.row_id,
                    "namespaces": list(row.namespaces),
                    "public_name_count": len(row.public_names),
                    "public_names": list(row.public_names),
                    "schema_ids": list(row.schema_ids),
                    "case_ids": list(row.case_ids),
                    "coverage_status": row.coverage_status,
                    "names_origin": row.names_origin,
                }
                for row in guarantees.rows
            ],
            "searched_item_count": len(items),
            "searched_items_by_kind": {
                kind: sum(1 for item in items if item.kind == kind)
                for kind in sorted({item.kind for item in items})
            },
        },
        "runtime_facts": runtime,
        "loader_index": {
            "parsed_library_files": loaders.parsed_files,
            "literals_with_dispatch_sites": len(loaders.literal_sites),
            "symbols_with_definition_sites": len(loaders.definition_sites),
            "environment_variables": sorted(loaders.environment_variables),
            "parse_failures": loaders.parse_failures[:20],
        },
        "corpora": [corpus.describe() for corpus in corpora],
        "control_corpus": control_corpus.describe(),
        "prefilter": prefilter_stats,
        "controls": controls,
        "control_records": [record.as_dict() for record in control_records],
        "records": [record.as_dict() for record in records],
        "record_count": len(records),
        "occurrence_count": sum(record.occurrences_in_file for record in records),
        "truncation": aggregator.truncation(),
        "row_verdicts": build_verdicts(guarantees, records, corpora, healthy, aggregator),
        "unattributed": {
            "note": (
                "Live library values that could not be attributed to exactly one guaranteed "
                "row. They are evidence of use, but they credit no row with consumption."
            ),
            "record_count": sum(1 for r in records if r.row_id == UNATTRIBUTED_ROW),
            "items": sorted({r.item for r in records if r.row_id == UNATTRIBUTED_ROW})[:60],
        },
        "diagnostics": sorted(set(analyzer.diagnostics))[:200],
        "diagnostic_count": len(set(analyzer.diagnostics)),
    }

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2, sort_keys=False) + "\n", encoding="utf-8")

    broken = [entry["channel"] for entry in controls if entry["status"] == "broken"]
    print(f"wrote {out_path}")
    print(
        f"records: {len(records)} "
        f"(occurrences {payload['occurrence_count']}, "
        f"truncated groups {payload['truncation']['groups_truncated']})"
    )
    for corpus in corpora:
        rows = payload["row_verdicts"][corpus.name]["rows"]
        consumed = sum(1 for row in rows.values() if row["verdict"] == "consumed")
        print(f"\n{corpus.name} ({corpus.role}): {consumed}/{len(guarantees.rows)} rows consumed")
        for row_id, row in rows.items():
            channels = ", ".join(row["channels"]) or "-"
            print(
                f"  {row_id:32s} {row['verdict']:24s} "
                f"files={row['kept_distinct_files']:<5d}+{row['omitted_file_hits']:<7d} "
                f"items={row['distinct_item_count']:<4d} "
                f"{channels}"
            )
    if broken:
        print(f"BROKEN CHANNELS (zeros withheld): {', '.join(broken)}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except ScanError as error:
        print(f"guarantee-dependency-scan: {error}", file=sys.stderr)
        raise SystemExit(3) from error
