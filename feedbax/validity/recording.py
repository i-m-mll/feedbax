"""Observe one managed worker process and stage the evidence of its deliveries."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import UTC, datetime
from functools import cached_property
from importlib import metadata
import os
from pathlib import Path
import platform
import sys
from threading import local
import time
from types import CodeType, ModuleType

from packaging.utils import canonicalize_name

from feedbax.contracts.artifact_custody import CONTENT_OBJECT_REF_PREFIX, ContentObjectRef
from feedbax.contracts.code_validity import AttemptProvenance, CodeTrace, FileSnapshot, ReceiptEvidence, SourcePath, SourceSnapshot
from feedbax.contracts.computation_validity import ComputationTrace
from feedbax.orchestration.revision import resolve_feedbax_provenance
from feedbax.persistence.artifact_custody import ImmutableArtifactBlobProvider
from feedbax.plugins.declarations import ResolvedDeclaration, observe_declarations
from feedbax.validity.environment import EnvironmentSnapshot, read_environment
from feedbax.validity.registry import registry_binding
from feedbax.validity.source import SourceCatalog, SourceFile, plugin_packages, source_catalog


type ComputationSnapshot = Callable[[], ComputationTrace | None]


@dataclass
class _Attempt:
    id: str
    store: ImmutableArtifactBlobProvider
    environment: EnvironmentSnapshot
    started: datetime
    excluded: tuple[Path, ...]
    resolutions: dict[tuple[str, str, int], ResolvedDeclaration] = field(default_factory=dict)
    files: dict[SourcePath, str] = field(default_factory=dict)
    indeterminate: set[SourcePath] = field(default_factory=set)
    indeterminate_files: set[SourcePath] = field(default_factory=set)
    prefixes: list[ContentObjectRef] = field(default_factory=list)
    computation: ComputationSnapshot | None = None

    def observe(self, family: str, key: str, definition: ResolvedDeclaration) -> None:
        self.resolutions[family, key, id(definition.defining_object)] = definition


_recorder: WorkerRecorder | None = None
_audit_installed = False


def _observe_open(event: str, arguments: tuple[object, ...]) -> None:
    if event == "open" and _recorder is not None:
        _recorder.observe_open(*arguments)


@dataclass(frozen=True)
class ExecutionDependencies:
    resolutions: tuple[tuple[str, str, ResolvedDeclaration], ...]
    files: tuple[tuple[SourcePath, str], ...]
    indeterminate_files: tuple[SourcePath, ...]


class WorkerRecorder:
    """Two lifetime rosters and a job-scoped audit over one worker process."""

    def __init__(self, project: Path):
        self.project = project.resolve()
        self.catalog = source_catalog(self.project)
        self.boot_ns = time.time_ns()
        self.entered: dict[str, set[str]] = defaultdict(set)
        self.host: dict[str, set[str]] = defaultdict(set)
        self.loaded: dict[SourcePath, SourceFile] = {}
        self.attempt: _Attempt | None = None
        self._audit = local()
        self._host_depth = 0

    @cached_property
    def _distribution_owners(self) -> dict[str, list[str]]:
        return metadata.packages_distributions()

    @cached_property
    def _interpreter_roots(self) -> tuple[Path, ...]:
        return tuple(dict.fromkeys((Path(sys.prefix).resolve(), Path(sys.base_prefix).resolve())))

    @contextmanager
    def monitor(self) -> Iterator[None]:
        global _recorder, _audit_installed
        if _recorder is not None:
            raise RuntimeError("A worker recorder already owns this process")
        monitor = sys.monitoring
        with_monitor = [ident for ident in (3, 4) if monitor.get_tool(ident) is None]
        if len(with_monitor) != 2:
            raise RuntimeError("Managed validity requires two free Python monitoring tool identifiers")
        self._entered_id, self._host_id = with_monitor
        monitor.use_tool_id(self._entered_id, "Feedbax executed source")
        monitor.use_tool_id(self._host_id, "Feedbax host product source")
        monitor.register_callback(self._entered_id, monitor.events.PY_START, self._enter)
        monitor.register_callback(self._host_id, monitor.events.PY_START, self._host_enter)
        monitor.set_events(self._entered_id, monitor.events.PY_START)
        _recorder = self
        if not _audit_installed:
            sys.addaudithook(_observe_open)
            _audit_installed = True
        try:
            yield
        finally:
            _recorder = None
            for identifier in with_monitor:
                monitor.set_events(identifier, 0)
                monitor.register_callback(identifier, monitor.events.PY_START, None)
                monitor.free_tool_id(identifier)

    def _enter(self, code: CodeType, _offset: int):
        self.entered[code.co_filename].add(code.co_qualname.replace(".<locals>.", "."))
        return sys.monitoring.DISABLE

    def _host_enter(self, code: CodeType, _offset: int):
        self.host[code.co_filename].add(code.co_qualname.replace(".<locals>.", "."))
        return sys.monitoring.DISABLE

    @contextmanager
    def host_product(self) -> Iterator[None]:
        if self._host_depth == 0:
            sys.monitoring.set_events(self._host_id, sys.monitoring.events.PY_START)
        self._host_depth += 1
        try:
            yield
        finally:
            self._host_depth -= 1
            if self._host_depth == 0:
                sys.monitoring.set_events(self._host_id, 0)

    def _source_paths(self) -> dict[SourcePath, tuple[set[str], set[str]]]:
        self.catalog = source_catalog(self.project)
        filenames = {
            name for module in tuple(sys.modules.values()) if isinstance(module, ModuleType)
            and isinstance(name := vars(module).get("__file__"), str)
        }
        filenames.update(name for name, entered in self.entered.items() if "<module>" in entered)
        result: dict[SourcePath, tuple[set[str], set[str]]] = {}
        for name in filenames:
            if not name.endswith(".py"):
                continue
            source = self.catalog.locate(Path(name))
            if source is None:
                continue
            if source.scope == "project" and any(Path(name).is_relative_to(root) for root in self._interpreter_roots):
                continue
            names, host = result.setdefault(source, (set(), set()))
            names.update(self.entered.get(name, ()))
            host.update(self.host.get(name, ()))
        return result

    def note_preloaded_sources(self) -> None:
        for source in self._source_paths():
            if (present := self.catalog.read(source)) is not None:
                self.loaded[source] = present

    def changed(self) -> bool:
        for source, expected in self.loaded.items():
            try:
                present = self.catalog.read(source)
            except OSError:
                return True
            if present is None or present.sha256 != expected.sha256:
                return True
        return False

    @contextmanager
    def delivery(self, job: str, store_root: Path, scratch: Path) -> Iterator[None]:
        environment = read_environment(self.project, verify_installation=True)
        store = ImmutableArtifactBlobProvider(store_root)
        excluded = (store_root.resolve(), scratch.resolve(), *self._interpreter_roots)
        with self._scope(_Attempt(job, store, environment, datetime.now(UTC), excluded)):
            yield

    @contextmanager
    def nested(self, attempt: str) -> Iterator[None]:
        parent = self.attempt
        assert parent is not None
        current = _Attempt(attempt, parent.store, parent.environment, datetime.now(UTC), parent.excluded)
        with self._scope(current):
            yield

    @contextmanager
    def _scope(self, attempt: _Attempt) -> Iterator[None]:
        previous, self.attempt = self.attempt, attempt
        try:
            with observe_declarations(attempt.observe):
                yield
        finally:
            self.attempt = previous

    def observe_open(self, filename: object, _mode: object, flags: object) -> None:
        attempt = self.attempt
        if (
            attempt is None or getattr(self._audit, "reading", False)
            or not isinstance(filename, (str, bytes)) or not isinstance(flags, int)
            or flags & os.O_ACCMODE == os.O_WRONLY
        ):
            return
        path = Path(os.fsdecode(filename))
        if path.suffix in {".py", ".pyc", ".pyo"}:
            return
        named = Path(os.path.abspath(path))
        if any(named.is_relative_to(root) for root in attempt.excluded):
            return
        try:
            source = self.catalog.locate(named)
            if source is None:
                return
            target = self.catalog.resolve(source)
            if target is None or any(target.is_relative_to(root) for root in attempt.excluded):
                return
        except (OSError, RuntimeError, ValueError):
            return
        self._audit.reading = True
        try:
            digest = self.catalog.file_digest(source)
        except OSError:
            attempt.indeterminate_files.add(source)
        else:
            if digest is not None and attempt.files.setdefault(source, digest) != digest:
                attempt.indeterminate_files.add(source)
        finally:
            self._audit.reading = False

    def snapshot(self, store: ImmutableArtifactBlobProvider) -> ReceiptEvidence:
        attempt = self.attempt
        assert attempt is not None
        if store.root != attempt.store.root:
            raise ValueError("An execution snapshot must use its attempt's Store")
        self._audit.reading = True
        try:
            return self._snapshot(attempt, store)
        finally:
            self._audit.reading = False

    def _snapshot(self, attempt: _Attempt, store: ImmutableArtifactBlobProvider) -> ReceiptEvidence:
        sources = []
        order = lambda source: (source.scope, source.path)
        for path, (entered, host) in sorted(self._source_paths().items(), key=lambda item: order(item[0])):
            source = self._current_source(path)
            if source is None:
                attempt.indeterminate.add(path)
                continue
            if source.modified_ns > self.boot_ns:
                attempt.indeterminate.add(path)
            ref = ContentObjectRef(
                object_id=CONTENT_OBJECT_REF_PREFIX + source.sha256,
                sha256=source.sha256, size_bytes=len(source.data),
            )
            try:
                with store.open_owned_object(ref):
                    pass
            except FileNotFoundError:
                store.store_object(source.data, expected=ref)
            sources.append(SourceSnapshot(
                path=path, content=ref, entered=tuple(sorted(entered)), host_product=tuple(sorted(host)),
            ))
        provenance = resolve_feedbax_provenance()
        for path, digest in attempt.files.items():
            try:
                stable = self.catalog.file_digest(path) == digest
            except OSError:
                stable = False
            if not stable:
                attempt.indeterminate_files.add(path)
        return ReceiptEvidence(
            attempt=attempt.id,
            trace=CodeTrace(
                sources=tuple(sources),
                registry=tuple(sorted({
                    registry_binding(family, key, value, self.catalog)
                    for (family, key, _), value in attempt.resolutions.items()
                }, key=lambda binding: (binding.family, binding.requested_key, binding.module, binding.qualified_name))),
                files=tuple(FileSnapshot(path=path, sha256=attempt.files[path]) for path in sorted(attempt.files, key=order)),
                environment=self._loaded_environment(attempt.environment),
                indeterminate=tuple(sorted(attempt.indeterminate, key=order)),
                indeterminate_files=tuple(sorted(attempt.indeterminate_files, key=order)),
                prefixes=tuple(attempt.prefixes),
                computation=attempt.computation() if attempt.computation is not None else None,
            ),
            provenance=AttemptProvenance(
                feedbax_commit=provenance.revision, feedbax_dirty=provenance.dirty,
                lock_sha256=attempt.environment.lock_sha256, platform=platform.platform(),
                started_at=attempt.started, ended_at=datetime.now(UTC),
            ),
        )

    def _current_source(self, path: SourcePath) -> SourceFile | None:
        try:
            present = self.catalog.read(path)
        except OSError:
            present = None
        if (captured := self.loaded.get(path)) is not None:
            if present is None or present.sha256 != captured.sha256:
                assert self.attempt is not None
                self.attempt.indeterminate.add(path)
            return captured
        if present is not None:
            self.loaded[path] = present
        return present

    def _loaded_environment(self, environment: EnvironmentSnapshot) -> dict[str, str]:
        first_party = {"feedbax", *plugin_packages().values()}
        loaded = {
            canonicalize_name(distribution)
            for name in {module.split(".", 1)[0] for module in tuple(sys.modules)}
            for distribution in self._distribution_owners.get(name, ())
        }
        return {name: environment.versions[name] for name in sorted(loaded - first_party) if name in environment.versions}


@contextmanager
def host_product() -> Iterator[None]:
    """Observe the Python code that contributes to an output outside a lowered program."""
    if _recorder is None:
        yield
    else:
        with _recorder.host_product():
            yield


@contextmanager
def nested_execution(attempt: str) -> Iterator[None]:
    """Scope an Evaluate point executing inside its producer's Train delivery."""
    if _recorder is None:
        yield
    else:
        with _recorder.nested(attempt):
            yield


def record_computation(snapshot: ComputationSnapshot) -> None:
    if _recorder is not None and (attempt := _recorder.attempt) is not None:
        attempt.computation = snapshot


def set_execution_prefix(ref: ContentObjectRef) -> None:
    if _recorder is not None and (attempt := _recorder.attempt) is not None and ref not in attempt.prefixes:
        attempt.prefixes.append(ref)


def snapshot_execution(store: ImmutableArtifactBlobProvider) -> ReceiptEvidence | None:
    if _recorder is None or _recorder.attempt is None:
        return None
    return _recorder.snapshot(store)


def execution_dependencies() -> ExecutionDependencies | None:
    """Detach observations needed to repeat an already prepared live operation."""
    if _recorder is None or (attempt := _recorder.attempt) is None:
        return None
    return ExecutionDependencies(
        tuple((family, key, resolved) for (family, key, _), resolved in attempt.resolutions.items()),
        tuple(attempt.files.items()), tuple(attempt.indeterminate_files),
    )


def inherit_execution_dependencies(dependencies: ExecutionDependencies | None) -> None:
    if dependencies is None or _recorder is None or (attempt := _recorder.attempt) is None:
        return
    for family, key, definition in dependencies.resolutions:
        attempt.observe(family, key, definition)
    attempt.indeterminate_files.update(dependencies.indeterminate_files)
    for source, digest in dependencies.files:
        if attempt.files.setdefault(source, digest) != digest:
            attempt.indeterminate_files.add(source)
