"""Traps the guarantee dependency scanner must not fall back into.

Each test here corresponds to a way a previous sweep got the wrong answer:
resolving a row to the empty list and calling it unconsumed, crediting a
downstream package's identically spelled constant to this library, believing a
name mentioned in prose, believing an allowlist that restates imports as data,
missing a symbol imported from a flattened re-export namespace, or manufacturing
consumers out of a sealed copy of this library's own source.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from typing import Any

import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "guarantee_dependency_scan.py"


def _load_scanner() -> Any:
    spec = importlib.util.spec_from_file_location("guarantee_dependency_scan", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    # Registered before execution because the module defines dataclasses, which
    # resolve their own module during class creation.
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def scanner() -> Any:
    return _load_scanner()


@pytest.fixture(scope="module")
def guarantees(scanner: Any) -> Any:
    return scanner.GuaranteeSet.load(
        ROOT / scanner.DEFAULT_MANIFEST, ROOT / scanner.DEFAULT_POLICY_DOC
    )


@pytest.fixture(scope="module")
def analyzer(scanner: Any, guarantees: Any) -> Any:
    """The same item set and loader index the program itself assembles."""
    runtime = scanner.collect_runtime_facts(ROOT)
    assert runtime["errors"] == []
    items, loaders = scanner.prepare_items_and_loaders(ROOT, guarantees, runtime)
    return scanner.Analyzer(
        guarantees=guarantees,
        items_by_value=scanner.index_items_by_value(items),
        names_by_namespace=scanner.index_names_by_namespace(items),
        loaders=loaders,
        max_structural_bytes=64 * 1024 * 1024,
    )


def _corpus(scanner: Any, root: Path, restatement: tuple[str, ...] = ()) -> Any:
    return scanner.Corpus(
        name="test",
        role="test",
        root=root,
        restatement_prefixes=restatement,
        restatement_reason=None,
    )


# ---------------------------------------------------------------------------
# The guarantee set itself
# ---------------------------------------------------------------------------


def test_manifest_deferred_rows_resolve_to_a_real_inventory(guarantees: Any) -> None:
    """The two rows whose document cell says "inventory below" are not empty.

    This is the failure that made a prior sweep report a heavily used row as
    unconsumed: it parsed only the Markdown, got the empty list for these rows,
    searched for nothing, and believed the result.
    """
    rows = guarantees.rows_by_id()
    for row_id in ("plugin-bootstrap", "figure-composition"):
        row = rows[row_id]
        assert row.names_origin == "manifest-inventory"
        assert len(row.public_names) > 5, row_id
    assert {
        "TRAINING_METHODS",
        "FamilyRequirement",
        "PluginRegistration",
        "RegistrationContext",
    } <= set(rows["plugin-bootstrap"].public_names)


def test_every_row_yields_something_to_search_for(guarantees: Any) -> None:
    for row in guarantees.rows:
        assert row.public_names, f"{row.row_id} has nothing to search for"
        assert row.namespaces, f"{row.row_id} has no namespace"


def test_empty_deferred_inventory_is_refused(scanner: Any, tmp_path: Path) -> None:
    """A row that defers to the manifest and finds nothing there must fail loudly."""
    manifest = json.loads((ROOT / scanner.DEFAULT_MANIFEST).read_text(encoding="utf-8"))
    for row in manifest["guaranteed_rows"]:
        if row["row_id"] == "plugin-bootstrap":
            row.pop("plugin_api")
    broken = tmp_path / "policy_manifest.v1.json"
    broken.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(scanner.ScanError) as caught:
        scanner.GuaranteeSet.load(broken, ROOT / scanner.DEFAULT_POLICY_DOC)
    assert "plugin-bootstrap" in str(caught.value)
    assert "unconsumed" in str(caught.value)


def test_row_set_disagreement_is_refused(scanner: Any, tmp_path: Path) -> None:
    manifest = json.loads((ROOT / scanner.DEFAULT_MANIFEST).read_text(encoding="utf-8"))
    manifest["guaranteed_rows"] = [
        row for row in manifest["guaranteed_rows"] if row["row_id"] != "graph-spec"
    ]
    broken = tmp_path / "policy_manifest.v1.json"
    broken.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(scanner.ScanError) as caught:
        scanner.GuaranteeSet.load(broken, ROOT / scanner.DEFAULT_POLICY_DOC)
    assert "disagree on the guaranteed row set" in str(caught.value)


# ---------------------------------------------------------------------------
# Known false positives
# ---------------------------------------------------------------------------


def test_a_downstream_packages_own_constant_is_not_a_dependency(
    scanner: Any, analyzer: Any, tmp_path: Path
) -> None:
    """Same spelling, different owner. `RESULT_SCHEMA_ID` is guaranteed by this
    library and also defined locally by a downstream package that never imports
    it. A definition is not a dependency."""
    (tmp_path / "post_run.py").write_text(
        'RESULT_SCHEMA_ID = "acme.analysis.thing"\n'
        'RESULT_SCHEMA_VERSION = "acme.analysis.thing.v1"\n'
        "REQUIRED_CASE_IDS = ()\n"
        "\n"
        "def payload():\n"
        '    return {"schema_id": RESULT_SCHEMA_ID, "schema_version": RESULT_SCHEMA_VERSION}\n',
        encoding="utf-8",
    )
    records = analyzer.analyze(_corpus(scanner, tmp_path), "post_run.py")
    live = [record for record in records if record.evidence_class == "dependency"]
    assert live == [], [record.as_dict() for record in live]


def test_a_name_inside_free_text_is_prose_not_a_dependency(
    scanner: Any, analyzer: Any, tmp_path: Path
) -> None:
    (tmp_path / "notes.json").write_text(
        json.dumps(
            {
                "notes": "We considered whether GraphSpec was the right shape and decided no.",
                "review": "The reviewer mentioned ComponentRegistry during the walkthrough.",
            }
        ),
        encoding="utf-8",
    )
    records = analyzer.analyze(_corpus(scanner, tmp_path), "notes.json")
    assert records, "the prose mention should still be recorded, just not as a dependency"
    assert all(record.evidence_class == "prose" for record in records)


def test_an_import_allowlist_is_a_restatement_not_a_dependency(
    scanner: Any, analyzer: Any, tmp_path: Path
) -> None:
    """A policy file that lists symbols as data restates a surface. It declares
    a rule about imports; it is not an independent consumer of them."""
    (tmp_path / "ci").mkdir()
    (tmp_path / "ci" / "surface.toml").write_text(
        "schema_version = 1\n"
        "[[allowed_file]]\n"
        'path = "src/example/method.py"\n'
        'symbols = ["GraphSpec", "TRAINING_METHODS", "RegistrationContext"]\n',
        encoding="utf-8",
    )
    corpus = _corpus(scanner, tmp_path, restatement=("ci/",))
    records = analyzer.analyze(corpus, "ci/surface.toml")
    assert records
    assert all(record.evidence_class == "restatement" for record in records)


# ---------------------------------------------------------------------------
# Known truths the scanner must reproduce
# ---------------------------------------------------------------------------


def test_a_symbol_imported_from_a_flattened_reexport_is_found(
    scanner: Any, analyzer: Any, tmp_path: Path
) -> None:
    """The row declares `feedbax.contracts.array_values`; the consumer imports
    from the flattened `feedbax.contracts`. A prior sweep searched only the
    declared sub-namespace and missed every one of these."""
    (tmp_path / "consumer.py").write_text(
        "from feedbax.contracts import (\n"
        "    ConstantArrayValueSpec,\n"
        "    SparseCooArrayValueSpec,\n"
        "    materialize_array_value,\n"
        ")\n",
        encoding="utf-8",
    )
    records = analyzer.analyze(_corpus(scanner, tmp_path), "consumer.py")
    found = {
        record.item
        for record in records
        if record.row_id == "array-values" and record.channel == "python-import"
    }
    assert found == {
        "ConstantArrayValueSpec",
        "SparseCooArrayValueSpec",
        "materialize_array_value",
    }
    detail = next(record.detail for record in records if record.item == "SparseCooArrayValueSpec")
    assert detail.startswith("from feedbax.contracts import SparseCooArrayValueSpec")


def test_an_import_from_an_undeclared_namespace_is_found_and_annotated(
    scanner: Any, analyzer: Any, tmp_path: Path
) -> None:
    """`figure-composition` declares `feedbax.contracts.figures`. An import of one
    of its names through a different namespace is still a dependency, and the
    record says so rather than quietly dropping it."""
    (tmp_path / "consumer.py").write_text(
        "from feedbax.contracts import FigureCompositionSpec\n", encoding="utf-8"
    )
    records = analyzer.analyze(_corpus(scanner, tmp_path), "consumer.py")
    hits = [record for record in records if record.item == "FigureCompositionSpec"]
    assert hits and hits[0].row_id == "figure-composition"
    assert "re-export namespace" in hits[0].detail


def test_aliased_module_attribute_use_is_found(scanner: Any, analyzer: Any, tmp_path: Path) -> None:
    (tmp_path / "consumer.py").write_text(
        "import feedbax.contracts.graph as g\n\n\ndef f():\n    return g.GraphSpec\n",
        encoding="utf-8",
    )
    records = analyzer.analyze(_corpus(scanner, tmp_path), "consumer.py")
    assert any(
        record.channel == "python-attribute" and record.item == "GraphSpec" for record in records
    )


def test_a_discriminator_value_is_found_at_any_depth_and_names_its_loader(
    scanner: Any, analyzer: Any, tmp_path: Path
) -> None:
    """`{"kind": "GraphSpec"}` binds a saved document to this library with no
    import and no schema string. The record must carry the JSON pointer and the
    library lines that branch on the literal."""
    (tmp_path / "receipt.json").write_text(
        json.dumps({"method_payload": {"payload": {"graph": {"kind": "GraphSpec"}}}}),
        encoding="utf-8",
    )
    records = analyzer.analyze(_corpus(scanner, tmp_path), "receipt.json")
    hits = [
        record
        for record in records
        if record.channel == "document-type-discriminator" and record.item == "GraphSpec"
    ]
    assert len(hits) == 1
    assert hits[0].json_pointer == "/method_payload/payload/graph/kind"
    assert hits[0].row_id == "graph-spec"
    loader = " ".join(hits[0].loader)
    assert "feedbax/contracts/manifest.py" in loader
    assert "feedbax/contracts/migrations.py" in loader


def test_an_older_live_schema_version_is_found_by_family(
    scanner: Any, analyzer: Any, tmp_path: Path
) -> None:
    (tmp_path / "saved.json").write_text(
        json.dumps({"schema_version": "feedbax.spec.graph.v1"}), encoding="utf-8"
    )
    records = analyzer.analyze(_corpus(scanner, tmp_path), "saved.json")
    assert any(
        record.channel == "schema-family-prefix" and record.row_id == "graph-spec"
        for record in records
    )


def test_pickled_class_identity_is_found(scanner: Any, analyzer: Any, tmp_path: Path) -> None:
    """This library deliberately preserves legacy class identity for decoding,
    so a saved pickle depends on the class path with no import and no literal."""
    (tmp_path / "state.pkl").write_bytes(b"cfeedbax.contracts.graph\nGraphSpec\n.")
    records = analyzer.analyze(_corpus(scanner, tmp_path), "state.pkl")
    assert any(
        record.channel == "pickle-class-identity" and record.item == "GraphSpec"
        for record in records
    )


# ---------------------------------------------------------------------------
# Corpus construction
# ---------------------------------------------------------------------------


def test_a_sealed_source_snapshot_is_excluded_but_its_sibling_data_is_not(
    scanner: Any, tmp_path: Path
) -> None:
    """An artifact tree holds verbatim copies of this library's source next to
    real run output. Excluding the whole tree loses the evidence; keeping the
    copies manufactures consumers. Only the copies go."""
    artifacts = tmp_path / "_artifacts"
    snapshot = artifacts / "run-1" / "repo-snapshots" / "feedbax"
    snapshot.mkdir(parents=True)
    (snapshot / "pyproject.toml").write_text('[project]\nname = "feedbax"\n', encoding="utf-8")
    (snapshot / "graph.py").write_text(
        "from feedbax.contracts.graph import GraphSpec\n", encoding="utf-8"
    )
    (snapshot / "fixture.json").write_text('{"kind": "GraphSpec"}', encoding="utf-8")
    output = artifacts / "run-1" / "collected"
    output.mkdir(parents=True)
    (output / "receipt.json").write_text('{"graph": {"kind": "GraphSpec"}}', encoding="utf-8")

    roots = scanner.detect_snapshot_roots(artifacts)
    assert roots == [snapshot.parent]

    enumerated = {
        str(path.relative_to(tmp_path)) for path in scanner.walk_artifact_root(artifacts, roots)
    }
    assert enumerated == {"_artifacts/run-1/collected/receipt.json"}


def test_a_snapshot_named_by_convention_without_a_build_file_is_excluded(
    scanner: Any, tmp_path: Path
) -> None:
    artifacts = tmp_path / "_artifacts"
    snapshot = artifacts / "run-1" / ".repo-snapshots" / "src"
    snapshot.mkdir(parents=True)
    (snapshot / "graph.py").write_text("GraphSpec = None\n", encoding="utf-8")
    (artifacts / "run-1" / "receipt.json").write_text("{}", encoding="utf-8")

    roots = scanner.detect_snapshot_roots(artifacts)
    assert roots == [artifacts / "run-1" / ".repo-snapshots"]
    enumerated = {
        str(path.relative_to(tmp_path)) for path in scanner.walk_artifact_root(artifacts, roots)
    }
    assert enumerated == {"_artifacts/run-1/receipt.json"}


# ---------------------------------------------------------------------------
# Controls
# ---------------------------------------------------------------------------


def test_every_channel_has_a_passing_positive_control(scanner: Any, analyzer: Any) -> None:
    """A channel that cannot find its own known-live example must never be
    allowed to report a zero elsewhere."""
    config = json.loads((ROOT / scanner.DEFAULT_CORPORA).read_text(encoding="utf-8"))
    corpus = scanner.build_corpus(config["control_corpus"], ROOT, scan_artifacts=True)
    records: list[Any] = []
    for relpath in corpus.all_files():
        records.extend(analyzer.analyze(corpus, relpath))
    controls = scanner.evaluate_controls(records)
    broken = [entry["channel"] for entry in controls if entry["status"] != "pass"]
    assert broken == []
    assert {entry["channel"] for entry in controls} == set(scanner.CHANNELS)


def test_control_corpus_negatives_produce_no_dependency_records(
    scanner: Any, analyzer: Any
) -> None:
    config = json.loads((ROOT / scanner.DEFAULT_CORPORA).read_text(encoding="utf-8"))
    corpus = scanner.build_corpus(config["control_corpus"], ROOT, scan_artifacts=True)
    leaks: list[Any] = []
    for relpath in corpus.all_files():
        if not relpath.startswith(("negative/", "restatement/")):
            continue
        leaks.extend(
            record
            for record in analyzer.analyze(corpus, relpath)
            if record.evidence_class == "dependency"
        )
    assert leaks == [], [record.as_dict() for record in leaks]


def test_bulk_artifact_evidence_cannot_evict_tracked_source_evidence(scanner: Any) -> None:
    """The aggregator caps files per item to keep the output readable. That cap
    must not spend its whole budget on run output and drop the one tracked file a
    reader would actually open: a receipt in `results/` was silently lost this
    way behind sixty thousand artifacts."""

    def record(path: str, tier: str) -> Any:
        return scanner.Record(
            corpus="c",
            role="active-consumer",
            row_id="graph-spec",
            item="GraphSpec",
            item_kind="public_name",
            channel="document-type-discriminator",
            path=path,
            line=None,
            json_pointer="/graph/kind",
            loader=(),
            loader_kind="",
            evidence_class="dependency",
            strength="direct",
            detail="",
            tier=tier,
        )

    aggregator = scanner.RecordAggregator(limit=10)
    for index in range(500):
        aggregator.add(record(f"_artifacts/run/{index}.json", "artifact"))
    aggregator.add(record("results/training/receipt.json", "tracked"))

    kept = {r.path for r in aggregator.records()}
    assert "results/training/receipt.json" in kept
    counts = aggregator.file_counts("c", "graph-spec", "dependency")
    assert counts["kept_distinct_files"] == 11
    assert counts["omitted_file_hits"] == 490


def test_corpus_enumerates_tracked_source_before_bulk_artifacts(scanner: Any) -> None:
    corpus = scanner.Corpus(
        name="c",
        role="active-consumer",
        root=Path("/nonexistent"),
        restatement_prefixes=(),
        restatement_reason=None,
        source_files=["src/b.py", "src/a.py"],
        artifact_files=["_artifacts/z.json", "_artifacts/a.json"],
    )
    assert corpus.all_files() == [
        "src/a.py",
        "src/b.py",
        "_artifacts/a.json",
        "_artifacts/z.json",
    ]
    assert corpus.tier("src/a.py") == "tracked"
    assert corpus.tier("_artifacts/a.json") == "artifact"
