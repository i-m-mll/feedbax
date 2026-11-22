"""Behavioral witnesses for managed receipt source and environment evidence."""

from datetime import UTC, datetime, timedelta
from hashlib import sha256
import importlib
import importlib.util
import os
from pathlib import Path
import re
import socket
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
from feedbax.contracts.store_closure import StoreClosureRef
from feedbax.host import worker
from feedbax.host.io import EVIDENCE_FILE, TOKEN_ENV, append, read_worker_result
from feedbax.host.source_roster import SourceRoster
from feedbax.host.worker import _record_execution
from feedbax.orchestration.revision import FeedbaxProvenance
from feedbax.persistence.artifact_custody import ImmutableArtifactBlobProvider
from feedbax.persistence.effective_success import SuccessView, store_request
from feedbax.persistence.reachability import measure_store
from feedbax.persistence.receipt_closure import ReceiptClosure
from feedbax.persistence.store_sync import sync_store
from feedbax.plugins.declarations import DeclarationRegistry, DistributionSource
from feedbax.validity.environment import EnvironmentMismatch, InstalledInventory, read_environment
from feedbax.persistence.evidence import read_execution_evidence
from feedbax.validity.recording import WorkerRecorder, checkpoint_execution_endpoint, numerical_work
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

    monkeypatch.syspath_prepend(str(project))
    deferred = project / "_managed_deferred_source.py"
    created = project / "_managed_created_source.py"

    def execute():
        monkeypatch.setitem(sys.modules, deferred.stem, None)
        monkeypatch.delitem(sys.modules, deferred.stem)
        assert importlib.import_module(deferred.stem).value == 2
        created.write_text("value = 3\n")
        monkeypatch.setitem(sys.modules, created.stem, None)
        monkeypatch.delitem(sys.modules, created.stem)
        assert importlib.import_module(created.stem).value == 3
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
            recorder.observe_new_sources()
            recorder.bind_project(project)
            registry.add(
                "example.operation", DistributionSource("example", "example", "1"), module.read_input,
                None, renamed_from=("example.read",),
            )
            recorder.start_delivery()
            with recorder.delivery("preparation", store.root, tmp_path / "scratch"):
                assert module.prepare() == 7

            deferred.write_text("value = 2\n")
            recorder.start_delivery()
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
    deferred_path = SourcePath(scope="project", path=deferred.name)
    created_path = SourcePath(scope="project", path=created.name)
    assert {deferred_path, created_path} <= {item.path for item in trace.sources}
    assert deferred_path not in trace.indeterminate
    assert created_path in trace.indeterminate
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


def test_reusable_worker_retires_after_edit_to_source_loaded_by_failed_delivery(tmp_path, monkeypatch):
    project = tmp_path / "project"
    project.mkdir()
    source = project / "_managed_failed_delivery.py"
    source.write_text("value = 1\n")
    monkeypatch.syspath_prepend(str(project))
    monkeypatch.setitem(sys.modules, source.stem, None)
    monkeypatch.delitem(sys.modules, source.stem)
    monkeypatch.setattr(
        "feedbax.validity.recording.resolve_feedbax_provenance",
        lambda: FeedbaxProvenance(Path("/example/feedbax"), "a" * 40, False),
    )
    monkeypatch.setattr(worker, "_redirect", lambda paths: None)
    monkeypatch.setattr(worker.os, "environ", {**os.environ, TOKEN_ENV: "worker-token"})

    def execute(_argv):
        assert importlib.import_module(source.stem).value == 1
        raise ValueError("Failed after importing project code")

    module = ModuleType("_managed_failure_entrypoint")
    module.main = execute
    monkeypatch.setitem(sys.modules, module.__name__, module)
    entrypoint = f"{module.__name__}:main"
    startup_log = tmp_path / "startup.log"
    startup_log.touch()

    def delivery(job):
        directory = tmp_path / job
        directory.mkdir()
        return {
            "job_id": job, "argv": ["python"], "entrypoint": entrypoint, "project_root": str(project),
            "store_root": str(tmp_path / "store"), "startup_log": str(startup_log),
            "paths": {name: str(directory / name) for name in (
                "event_log", "stdout", "stderr", "input", "stop_file", "scratch",
            )},
        }

    first = delivery("first")
    next_job = delivery("next")
    messages = iter((first, next_job))

    def receive_next(_control, _limit):
        message = next(messages)
        if message is next_job:
            assert read_worker_result(Path(first["paths"]["event_log"]))["error"]["code"] == "worker_exception"
            before = source.stat()
            source.write_text("value = 2\n")
            os.utime(source, ns=(before.st_atime_ns, before.st_mtime_ns))
        return message

    monkeypatch.setattr(worker, "receive", receive_next)
    control, peer = socket.socketpair()
    with peer, (tmp_path / "reports").open("wb") as reports, SourceRoster(3, "test worker source") as roster:
        roster.enable()
        recorder = WorkerRecorder(roster.names, boot_ns=time.time_ns())
        with recorder.monitor():
            assert worker._serve(
                {"preload": [], "warm_entrypoint": entrypoint, "light_affinity": ["project", "store"]},
                control, reports.fileno(), recorder,
            ) == 0
    assert not Path(next_job["paths"]["event_log"]).exists()


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
    site = tmp_path / "site"
    distribution = site / "example_lib-1.dist-info"
    distribution.mkdir(parents=True)
    metadata_file = distribution / "METADATA"
    metadata_file.write_text("Name: example-lib\nVersion: 1\n")
    monkeypatch.setattr("feedbax.validity.recording.InstalledInventory", lambda: InstalledInventory([str(site)]))
    lock = tmp_path / "uv.lock"
    lock.write_text('version = 1\n[[package]]\nname = "example-lib"\nversion = "1"\n')
    authored = tmp_path / "staged-project"
    authored.mkdir()
    recorder = WorkerRecorder({}, boot_ns=time.time_ns())
    recorder.bind_project(authored, environment_root=tmp_path)
    assert not recorder.changed()

    store = ImmutableArtifactBlobProvider(tmp_path / "store")
    with recorder.delivery("first", store.root, tmp_path / "scratch"):
        assert recorder.checkpoint(store, 1).completed_updates == 1
        before = metadata_file.stat()
        metadata_file.write_text("Name: example-lib\nVersion: 2\n")
        os.utime(metadata_file, ns=(before.st_atime_ns, before.st_mtime_ns))
        installed = {"example-lib": "2"}
        lock.write_text('version = 1\n[[package]]\nname = "example-lib"\nversion = "2"\n')
        assert read_environment(tmp_path, installed=installed, verify_installation=True).versions == installed
        with pytest.raises(EnvironmentMismatch) as mismatch:
            recorder.checkpoint(store, 2)
        assert {"example-lib", "1", "2"} <= set(re.findall(r"[\w-]+", str(mismatch.value)))

    assert recorder.changed()


def test_installed_inventory_detects_distribution_membership_and_replacement(tmp_path):
    site = tmp_path / "site"
    site.mkdir()
    installed = InstalledInventory([str(site)])
    assert installed.versions() == {}

    distribution = site / "example_lib-1.dist-info"
    distribution.mkdir()
    metadata_file = distribution / "METADATA"
    metadata_file.write_text("Name: example-lib\nVersion: 1\n")
    assert installed.versions() == {"example-lib": "1"}

    replacement = tmp_path / "replacement"
    replacement.write_text("Name: example-lib\nVersion: 2\n")
    prior = metadata_file.stat()
    os.utime(replacement, ns=(prior.st_atime_ns, prior.st_mtime_ns))
    replacement.replace(metadata_file)
    assert installed.versions() == {"example-lib": "2"}

    membership = site.stat()
    distribution.rename(tmp_path / distribution.name)
    os.utime(site, ns=(membership.st_atime_ns, membership.st_mtime_ns))
    assert installed.versions() == {}


def test_checkpoint_custody_carries_incremental_observations_to_another_store(tmp_path, monkeypatch):
    project = tmp_path / "project"
    project.mkdir()
    source = project / "operation.py"
    original = "def first(): return 1\ndef later(path): return path.read_text()\n"
    source.write_text(original)
    input_path = project / "parameters.txt"
    input_path.write_text("parameters")
    store = ImmutableArtifactBlobProvider(tmp_path / "store")
    monkeypatch.setattr(
        "feedbax.validity.recording.resolve_feedbax_provenance",
        lambda: FeedbaxProvenance(Path("/example/feedbax"), "a" * 40, False),
    )
    spec = importlib.util.spec_from_file_location("_checkpoint_trace_test_operation", source)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, spec.name, module)
    registry = DeclarationRegistry("operation")
    source_path = SourcePath(scope="project", path=source.name)

    with SourceRoster(3, "test checkpoint worker source") as roster:
        roster.enable()
        recorder = WorkerRecorder(roster.names, boot_ns=time.time_ns())
        with recorder.monitor():
            spec.loader.exec_module(module)
            recorder.bind_project(project)
            registry.add("example.later", DistributionSource("example", "example", "1"), module.later, None)
            with recorder.delivery("attempt", store.root, tmp_path / "scratch"):
                assert module.first() == 1
                first = checkpoint_execution_endpoint(store, 1)
                assert first is not None
                baseline = read_execution_evidence(store, first).trace
                source.write_text(original.replace("return 1", "return 2"))
                assert registry.resolve("example.later").defining_object(input_path) == "parameters"
                second = checkpoint_execution_endpoint(store, 2)
                assert second is not None
                source.write_text(original)
                final = checkpoint_execution_endpoint(store, 3)
                assert final is not None

    first_source = next(item for item in baseline.sources if item.path == source_path)
    assert "later" not in first_source.entered
    assert source_path not in baseline.indeterminate
    assert baseline.files == () and baseline.registry == ()
    closure = StoreClosureRef(roots=(final,))
    destination = ImmutableArtifactBlobProvider(tmp_path / "other-store")
    sync_store(store, destination, closure, destination_ref="checkpoint")
    assert destination.store_id != str(final.store_id)

    trace = read_execution_evidence(destination, final).trace
    final_source = next(item for item in trace.sources if item.path == source_path)
    assert {"first", "later"} <= set(final_source.entered)
    assert final_source.content == first_source.content
    assert destination.get_object(final_source.content) == original.encode()
    assert {item.requested_key for item in trace.registry} == {"example.later"}
    assert {(item.path.path, item.sha256) for item in trace.files} == {
        (input_path.name, sha256(b"parameters").hexdigest()),
    }
    assert source_path in trace.indeterminate
    assert destination.read_ref("checkpoint") == closure.model_dump(mode="json")
    assert store.compare_and_swap_ref("checkpoint", expected=None, replacement=closure.model_dump(mode="json"))
    reachability = measure_store(store, now=datetime.now(UTC) + timedelta(days=2), keep_superseded_days=0)
    assert reachability.eligible_digests is not None
    assert {first.evidence.sha256, second.evidence.sha256} <= set(reachability.eligible_digests)
    assert not {final.evidence.sha256, final_source.content.sha256} & set(reachability.eligible_digests)
    with pytest.raises(ValueError, match="Store, attempt or coordinate"):
        read_execution_evidence(destination, final.model_copy(update={"completed_updates": 2}))


def test_recorder_follows_each_delivery_project_until_it_loads_project_code(tmp_path, monkeypatch):
    first, second = tmp_path / "first", tmp_path / "second"
    for project in (first, second):
        project.mkdir()
    (second / "_held_project_source.py").write_text("value = 1\n")
    recorder = WorkerRecorder({}, boot_ns=time.time_ns())
    recorder.bind_project(first)
    recorder.bind_project(second)
    assert recorder.held_project is None
    monkeypatch.syspath_prepend(str(second))
    monkeypatch.setitem(sys.modules, "_held_project_source", None)
    monkeypatch.delitem(sys.modules, "_held_project_source")
    importlib.import_module("_held_project_source")
    recorder.observe_new_sources()
    assert recorder.held_project == second.resolve()
    recorder.bind_project(second)
    with pytest.raises(ValueError, match="loaded project code"):
        recorder.bind_project(first)
