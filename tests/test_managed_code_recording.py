"""Behavioral witnesses for managed receipt source and environment evidence."""

from hashlib import sha256
import importlib.util
from pathlib import Path
import re
import sys
import time
from types import ModuleType

import pytest

from feedbax.contracts.artifact_custody import ContentObjectRef
from feedbax.contracts.code_validity import SourcePath
from feedbax.contracts.effective_success import Product
from feedbax.contracts.report_document import ReportDocument, ReportText
from feedbax.contracts.request_identity import encode_request
from feedbax.contracts.request_refs import ContentRef
from feedbax.contracts.request_specs import ReportRequestSpec
from feedbax.host.io import EVIDENCE_FILE, append
from feedbax.host.source_roster import SourceRoster
from feedbax.host.worker import _record_execution
from feedbax.orchestration.revision import FeedbaxProvenance
from feedbax.persistence.artifact_custody import ImmutableArtifactBlobProvider
from feedbax.persistence.effective_success import SuccessView, store_request
from feedbax.persistence.receipt_closure import ReceiptClosure
from feedbax.plugins.declarations import DeclarationRegistry, DistributionSource
from feedbax.validity.environment import EnvironmentMismatch, read_environment
from feedbax.validity.recording import WorkerRecorder, numerical_work
from feedbax.validity.source import SourceCatalog, source_catalog


def test_source_comparison_counts_new_names_and_only_observed_function_bodies(tmp_path):
    source = tmp_path / "operation.py"
    source.write_text("def used():\n    return len([1])\n\ndef skipped():\n    return 1\n")
    path = SourcePath(scope="project", path=source.name)
    catalog = SourceCatalog({"project": tmp_path})
    original = catalog.read(path)
    assert original is not None

    def compare():
        return catalog.compare_source(path, original.sha256, lambda: original.data, ("used",))

    source.write_text(
        'def used():\n    """Count the samples."""\n    return len( [1] )\n'
        "\ndef skipped():\n    return 27\n"
    )
    assert compare() is None

    with source.open("a") as stream:
        stream.write("\ndef len(values):\n    return 7\n")
    change = compare()
    assert change is not None and change.statements == ("len",)

    source.write_text("def used():\n    return 5\n\ndef skipped():\n    return 27\n")
    change = compare()
    assert change is not None and change.functions == ("used",)


def test_dotted_plugin_declaration_tracks_helpers_elsewhere_in_its_package(tmp_path, monkeypatch):
    package = tmp_path / "installed" / "example"
    declarations = package / "declarations"
    declarations.mkdir(parents=True)
    helper = package / "numerics.py"
    helper.write_text("def run(): return 1\n")
    module = ModuleType("example.declarations")
    module.__file__ = str(declarations / "__init__.py")
    monkeypatch.setitem(sys.modules, module.__name__, module)
    monkeypatch.setattr("feedbax.validity.source.plugin_packages", lambda: {module.__name__: "example"})
    catalog = source_catalog(tmp_path / "project")
    path = catalog.locate(helper)
    assert path == SourcePath(scope="plugin:example", path="numerics.py")
    assert catalog.read(path) is not None

    extra_package = tmp_path / "editable" / "example"
    extra_declarations = extra_package / "declarations"
    extra_declarations.mkdir(parents=True)
    extra = extra_package / "numerics.py"
    extra.write_text("def run(): return 2\n")
    module.__file__ = None
    module.__path__ = [str(declarations), str(extra_declarations)]
    assert source_catalog(tmp_path / "project").locate(extra) == SourcePath(
        scope="plugin:example:1", path="numerics.py",
    )


def test_warm_job_retains_observed_source_registry_and_file_identity(tmp_path, monkeypatch):
    project = tmp_path / "project"
    project.mkdir()
    source = project / "operation.py"
    source.write_text(
        "def prepare():\n    return 7\n\ndef read_input(path):\n    return path.read_text()\n"
        "\ndef compute():\n    return 42\n\ndef run(argv):\n    return argv[0]()\n"
    )
    before = project / "before.txt"
    before.write_text("first contents")
    after = project / "after.txt"
    after.write_text("second contents")
    selected = project / "selected.txt"
    selected.symlink_to(before.name)
    store = ImmutableArtifactBlobProvider(tmp_path / "store")
    monkeypatch.setattr(
        "feedbax.validity.recording.resolve_feedbax_provenance",
        lambda: FeedbaxProvenance(Path("/example/feedbax"), "a" * 40, False),
    )
    spec = importlib.util.spec_from_file_location("_managed_trace_test_operation", source)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, spec.name, module)
    registry = DeclarationRegistry("operation")
    named_file = SourcePath(scope="project", path=selected.name)
    log = tmp_path / "worker" / "events.jsonl"
    log.parent.mkdir()
    log.touch()

    def execute():
        read = registry.resolve("example.read").defining_object
        assert read(selected) == "first contents"
        selected.unlink()
        selected.symlink_to(after.name)
        assert read(selected) == "second contents"
        with numerical_work():
            assert module.compute() == 42
        append(log, "result", {"outcome": "succeeded"})
        return 0

    boot_ns = time.time_ns()
    with SourceRoster(3, "test managed worker source") as roster:
        roster.enable()
        recorder = WorkerRecorder(roster.names, boot_ns=boot_ns)
        with recorder.monitor():
            spec.loader.exec_module(module)
            recorder.note_preloaded_sources()
            recorder.bind_project(project)
            registry.add(
                "example.operation", DistributionSource("example", "example", "1"), module.read_input,
                None, renamed_from=("example.read",),
            )
            with recorder.delivery("preparation", store.root, tmp_path / "scratch"):
                assert module.prepare() == 7

            assert _record_execution(
                {"job_id": "final", "store_root": str(store.root), "project_root": str(project),
                 "entrypoint": f"{spec.name}:run"},
                recorder, log, module.run, [execute],
            ) == 0
            stage = ContentObjectRef.model_validate_json(log.with_name(EVIDENCE_FILE).read_bytes())

            source.write_text("def prepare():\n    return 8\n")
            assert recorder.changed()

    report = ReportDocument(title="Example report", blocks=(ReportText(markdown="Observed values"),))
    request = store_request(store, encode_request(ReportRequestSpec(document=report), {}))
    blob = store.store_object(b"report")
    product = Product(name="report", content=ContentRef(sha256=blob.sha256, size=blob.size_bytes))
    success = SuccessView(store).publish_bound(
        request, attempt="final", products={"report": (product,)}, bindings={}, execution_evidence=stage,
    )
    trace = success.receipt.trace
    assert trace is not None
    captured = next(item for item in trace.sources if item.path == SourcePath(scope="project", path=source.name))
    assert "prepare" in captured.entered and "prepare" in captured.host_product
    assert {"run", "read_input"} <= set(captured.host_product)
    assert "compute" in captured.entered and "compute" not in captured.host_product
    runner = next(item for item in trace.sources if item.path == SourcePath(scope="feedbax", path="host/worker.py"))
    assert "_record_execution" in runner.host_product
    custody = next(item for item in trace.sources if item.path == SourcePath(scope="feedbax", path="validity/evidence.py"))
    assert "stage_evidence" in custody.host_product
    (registered,) = trace.registry
    assert (registered.requested_key, registered.key, registered.source_path) == (
        "example.read", "example.operation", captured.path,
    )
    assert {(item.path, item.sha256) for item in trace.files} == {(named_file, sha256(b"first contents").hexdigest())}
    assert trace.indeterminate_files == (named_file,)
    assert captured.content in ReceiptClosure(store).owned_objects(success.ref)
    assert b"return 7" in store.get_object(captured.content)
    assert recorder.catalog.file_digest(named_file) == sha256(b"second contents").hexdigest()


def test_worker_environment_verifies_versions_from_the_active_lock_projection(tmp_path):
    (tmp_path / "uv.lock").write_text(
        'version = 1\n[[package]]\nname = "example-lib"\nversion = "1"\n'
        'resolution-markers = ["python_version >= \'3.12\'"]\n'
        '[[package]]\nname = "example-lib"\nversion = "0.9"\n'
        'resolution-markers = ["python_version < \'3.12\'"]\n'
    )
    current = read_environment(tmp_path, installed={"example-lib": "2"})
    assert current.versions == {"example-lib": "1"}
    assert current.lock_sha256 is not None
    with pytest.raises(EnvironmentMismatch, match="example-lib 2"):
        read_environment(tmp_path, installed={"example-lib": "2"}, verify_installation=True)


def test_warm_worker_retires_when_a_dependency_and_lock_upgrade_together(tmp_path, monkeypatch):
    installed = {"example-lib": "1"}
    monkeypatch.setattr("feedbax.validity.recording.installed_versions", lambda: installed.copy())
    lock = tmp_path / "uv.lock"
    lock.write_text('version = 1\n[[package]]\nname = "example-lib"\nversion = "1"\n')
    authored = tmp_path / "staged-project"
    authored.mkdir()
    recorder = WorkerRecorder({}, boot_ns=time.time_ns())
    recorder.bind_project(authored, environment_root=tmp_path)
    assert not recorder.changed()

    with recorder.delivery("first", tmp_path / "store", tmp_path / "scratch"):
        installed["example-lib"] = "2"
        lock.write_text('version = 1\n[[package]]\nname = "example-lib"\nversion = "2"\n')
        assert read_environment(tmp_path, installed=installed, verify_installation=True).versions == installed
        with pytest.raises(EnvironmentMismatch) as mismatch:
            recorder.ensure_unchanged()
        assert {"example-lib", "1", "2"} <= set(re.findall(r"[\w-]+", str(mismatch.value)))

    assert recorder.changed()
