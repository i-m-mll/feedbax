"""Observe one managed worker process and stage the evidence of its deliveries."""

from __future__ import annotations

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
from time import time_ns
from types import ModuleType
from uuid import UUID

from packaging.utils import canonicalize_name

from feedbax.contracts.artifact_custody import CONTENT_OBJECT_REF_PREFIX, ContentObjectRef
from feedbax.contracts.canonical_json import canonical_json_v2_bytes
from feedbax.contracts.code_validity import (
    AttemptProvenance, CheckpointComputation, CheckpointEvidence, CheckpointEvidenceRoot, CodeTrace,
    ComputationUnavailable, ExecutionEvidenceRef, FileSnapshot, ObservationGeneration,
    ReceiptEvidence, SourcePath, SourceSnapshot,
)
from feedbax.contracts.computation_validity import ComputationPrograms, ComputationTrace
from feedbax.host.source_roster import SourceRoster
from feedbax.orchestration.revision import resolve_feedbax_provenance
from feedbax.persistence.artifact_custody import ImmutableArtifactBlobProvider
from feedbax.plugins.declarations import ResolvedDeclaration, observe_declarations
from feedbax.validity.environment import EnvironmentMismatch, EnvironmentSnapshot, InstalledInventory, read_environment
from feedbax.validity.registry import registry_binding
from feedbax.validity.source import SourceFile, plugin_packages, source_catalog


type ComputationSnapshot = Callable[[], ComputationTrace | ComputationUnavailable | None]


def _source_key(source: SourcePath) -> tuple[str, str]:
    return source.scope, source.path


@dataclass
class _Attempt:
    id: str
    store: ImmutableArtifactBlobProvider
    environment: EnvironmentSnapshot
    started: datetime
    excluded: tuple[Path, ...]
    resolutions: dict[tuple[str, str, int], ResolvedDeclaration] = field(default_factory=dict)
    files: dict[SourcePath, str] = field(default_factory=dict)
    indeterminate_files: set[SourcePath] = field(default_factory=set)
    prefixes: list[ExecutionEvidenceRef] = field(default_factory=list)
    computation: ComputationSnapshot | None = None
    store_id: UUID | None = None
    generation: ContentObjectRef | None = None
    generation_index: int = -1
    previous_sources: dict[SourcePath, SourceSnapshot] = field(default_factory=dict)
    previous_observations: dict = field(default_factory=dict)
    programs: tuple[ComputationPrograms, ContentObjectRef] | None = None

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

    def __init__(self, entered: dict[str, set[str]], *, boot_ns: int):
        self.project: Path | None = None
        self._environment_root: Path | None = None
        self.catalog = source_catalog(None)
        self._delivery_started_ns = boot_ns
        self._source_witnesses: dict[SourcePath, int] = {}
        self.installation = InstalledInventory()
        self.installed = self.installation.versions(fresh=True)
        self.feedbax = resolve_feedbax_provenance()
        self.entered = entered
        self.host = SourceRoster(4, "Feedbax host product source")
        self._source_roots: dict[str, Path] = {}
        self._source_names: dict[str, SourcePath | None] = {}
        self._archived: dict[Path, tuple[int, ...]] = {}
        self.loaded: dict[SourcePath, SourceFile] = {}
        self._source_records: dict[SourcePath, SourceSnapshot] = {}
        self._indeterminate: set[SourcePath] = set()
        self.attempt: _Attempt | None = None
        self._audit = local()
        self._numerical_depth = 0

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
        _recorder = self
        if not _audit_installed:
            sys.addaudithook(_observe_open)
            _audit_installed = True
        try:
            with self.host:
                self.host.enable()
                yield
        finally:
            _recorder = None

    @contextmanager
    def numerical_work(self) -> Iterator[None]:
        if self._numerical_depth == 0:
            self.host.disable()
        self._numerical_depth += 1
        try:
            yield
        finally:
            self._numerical_depth -= 1
            if self._numerical_depth == 0:
                self.host.enable()

    def _source_paths(self) -> dict[SourcePath, tuple[set[str], set[str]]]:
        self.catalog = source_catalog(self.project)
        if self._source_roots != self.catalog.roots:
            self._source_roots = self.catalog.roots.copy()
            self._source_names.clear()
        filenames = {
            name for module in tuple(sys.modules.values()) if isinstance(module, ModuleType)
            and isinstance(name := vars(module).get("__file__"), str)
        }
        filenames.update(name for name, entered in self.entered.items() if "<module>" in entered)
        result: dict[SourcePath, tuple[set[str], set[str]]] = {}
        for name in filenames:
            if not name.endswith(".py"):
                continue
            if name not in self._source_names:
                named = Path(name)
                source = self.catalog.locate(named)
                if source is not None and source.scope == "project" and any(
                    named.resolve().is_relative_to(root) for root in self._interpreter_roots
                ):
                    source = None
                self._source_names[name] = source
            source = self._source_names[name]
            if source is None:
                continue
            names, host = result.setdefault(source, (set(), set()))
            names.update(self.entered.get(name, ()))
            host.update(self.host.names.get(name, ()))
        return result

    def note_preloaded_sources(self) -> None:
        for path in self._source_paths():
            if path not in self.loaded:
                self._current_source(path)

    def start_delivery(self) -> None:
        started = time_ns()
        self.note_preloaded_sources()
        self._delivery_started_ns = started

    def bind_project(self, project: Path, *, environment_root: Path | None = None) -> None:
        root = project.resolve(strict=True)
        lock_root = environment_root.resolve(strict=True) if environment_root is not None else root
        if self.project is None:
            self.project = root
            self._environment_root = lock_root
            self.note_preloaded_sources()
        elif self.project != root or self._environment_root != lock_root:
            raise ValueError("A managed worker cannot change its project or environment authority")

    def changed(self) -> bool:
        if self.installation.versions() != self.installed:
            return True
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
        environment = self._current_environment(fresh=True)
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

    @contextmanager
    def _capture(self, store: ImmutableArtifactBlobProvider) -> Iterator[_Attempt]:
        attempt = self.attempt
        assert attempt is not None
        if store.root != attempt.store.root:
            raise ValueError("An execution snapshot must use its attempt's Store")
        self._audit.reading = True
        try:
            yield attempt
        finally:
            self._audit.reading = False

    def snapshot(self, store: ImmutableArtifactBlobProvider) -> ReceiptEvidence:
        with self._capture(store) as attempt:
            sources, observations, computation, provenance = self._observe(attempt, store)
            return ReceiptEvidence(
                attempt=attempt.id, provenance=provenance,
                trace=CodeTrace(
                    sources=tuple(sources.values()), **observations, prefixes=tuple(attempt.prefixes),
                    computation=computation if isinstance(computation, ComputationTrace) else None,
                    computation_unavailable=computation if isinstance(computation, ComputationUnavailable) else None,
                ),
            )

    def checkpoint(self, store: ImmutableArtifactBlobProvider, completed_updates: int) -> CheckpointEvidenceRoot:
        with self._capture(store) as attempt:
            sources, observations, computation, provenance = self._observe(attempt, store)
            if attempt.store_id is None:
                attempt.store_id = UUID(store.store_id)
            changes = tuple(
                source for path, source in sources.items() if attempt.previous_sources.get(path) is not source
            )
            changed_observations = {
                name: value for name, value in observations.items() if attempt.previous_observations.get(name) != value
            }
            generation = attempt.generation
            next_index = attempt.generation_index
            if generation is None or changes or changed_observations:
                next_index += 1
                record = ObservationGeneration(
                    store_id=attempt.store_id, attempt=attempt.id, index=next_index,
                    previous=generation, sources=changes, **changed_observations,
                )
                generation = store.store_object(canonical_json_v2_bytes(record.model_dump(mode="json")))
            if isinstance(computation, ComputationTrace):
                programs = ComputationPrograms(
                    normalization_version=computation.normalization_version,
                    reference_platform=computation.reference_platform, driver=computation.driver,
                    starts=computation.starts, programs=computation.programs,
                )
                if attempt.programs is None or attempt.programs[0] != programs:
                    program_ref = store.store_object(canonical_json_v2_bytes(programs.model_dump(mode="json")))
                    attempt.programs = programs, program_ref
                proof = CheckpointComputation(
                    programs=attempt.programs[1], completed_steps=computation.completed_steps, prefix=computation.prefix,
                )
            else:
                proof = computation
            endpoint = CheckpointEvidence(
                store_id=attempt.store_id, attempt=attempt.id, completed_updates=completed_updates,
                generation=generation, provenance=provenance, prefixes=tuple(attempt.prefixes), computation=proof,
            )
            ref = store.store_object(canonical_json_v2_bytes(endpoint.model_dump(mode="json")))
            attempt.generation, attempt.generation_index = generation, next_index
            attempt.previous_sources, attempt.previous_observations = sources, observations
            return CheckpointEvidenceRoot(
                evidence=ref, store_id=attempt.store_id, attempt=attempt.id, completed_updates=completed_updates,
            )

    def _observe(self, attempt: _Attempt, store: ImmutableArtifactBlobProvider):
        self._verify_environment(attempt)
        computation = attempt.computation() if attempt.computation is not None else None

        initial_roster = self._source_paths()
        archived = {}
        for path in sorted(initial_roster.keys() | self.loaded.keys(), key=_source_key):
            source = self._current_source(path)
            if source is None:
                continue
            previous = self._source_records.get(path)
            ref = previous.content if previous is not None else ContentObjectRef(
                object_id=CONTENT_OBJECT_REF_PREFIX + source.sha256,
                sha256=source.sha256, size_bytes=len(source.data),
            )
            if attempt.generation is None or path not in attempt.previous_sources:
                self._archive_source(store, ref, source.data)
            archived[path] = ref
        self._verify_file_reads(attempt)
        roster = self._source_paths()
        sources = {}
        for path, ref in archived.items():
            entered, host = roster.get(path, initial_roster.get(path, ((), ())))
            previous = self._source_records.get(path)
            if previous is not None:
                entered = set(entered).union(previous.entered)
                host = set(host).union(previous.host_product)
                if len(entered) == len(previous.entered) and len(host) == len(previous.host_product):
                    sources[path] = previous
                    continue
            record = SourceSnapshot(path=path, content=ref, entered=tuple(sorted(entered)), host_product=tuple(sorted(host)))
            sources[path] = self._source_records[path] = record
        observations = {
            "registry": tuple(sorted({
                registry_binding(family, key, value, self.catalog)
                for (family, key, _), value in attempt.resolutions.items()
            }, key=lambda binding: (binding.family, binding.requested_key, binding.module, binding.qualified_name))),
            "files": tuple(FileSnapshot(path=path, sha256=attempt.files[path]) for path in sorted(attempt.files, key=_source_key)),
            "environment": self._loaded_environment(attempt.environment),
            "indeterminate": tuple(sorted(self._indeterminate, key=_source_key)),
            "indeterminate_files": tuple(sorted(attempt.indeterminate_files, key=_source_key)),
        }
        provenance = AttemptProvenance(
            feedbax_commit=self.feedbax.revision, feedbax_dirty=self.feedbax.dirty,
            lock_sha256=attempt.environment.lock_sha256, platform=platform.platform(),
            started_at=attempt.started, ended_at=datetime.now(UTC),
        )
        return sources, observations, computation, provenance

    def _archive_source(self, store: ImmutableArtifactBlobProvider, ref: ContentObjectRef, data: bytes) -> None:
        path = store.object_path(ref)

        def occurrence() -> tuple[int, ...]:
            info = path.lstat()
            return info.st_dev, info.st_ino, info.st_size, info.st_mode, info.st_nlink, info.st_mtime_ns, info.st_ctime_ns

        try:
            current = occurrence()
            if self._archived.get(path) == current:
                return
            with store.open_owned_object(ref):
                self._archived[path] = occurrence()
        except FileNotFoundError:
            store.store_object(data, expected=ref)
            self._archived[path] = occurrence()

    def _current_source(self, path: SourcePath) -> SourceFile | None:
        witness = self._source_witnesses.setdefault(path, self._delivery_started_ns)
        try:
            present = self.catalog.read(path)
        except OSError:
            present = None
        captured = self.loaded.get(path)
        if present is None or present.modified_ns > witness or (
            captured is not None and present.sha256 != captured.sha256
        ):
            self._indeterminate.add(path)
        if captured is None and present is not None:
            self.loaded[path] = present
        return captured if captured is not None else present

    def _verify_file_reads(self, attempt: _Attempt) -> None:
        for path, digest in attempt.files.items():
            try:
                stable = self.catalog.file_digest(path) == digest
            except OSError:
                stable = False
            if not stable:
                attempt.indeterminate_files.add(path)

    def _current_environment(self, *, fresh: bool = False) -> EnvironmentSnapshot:
        assert self._environment_root is not None
        installed = self.installation.versions(fresh=fresh)
        if installed != self.installed:
            changed = [
                f"{name} ({self.installed.get(name, 'absent')} to {installed.get(name, 'absent')})"
                for name in sorted(self.installed.keys() | installed.keys())
                if self.installed.get(name) != installed.get(name)
            ]
            raise EnvironmentMismatch("Installed distributions changed in the live worker: " + ", ".join(changed))
        return read_environment(self._environment_root, installed=installed, verify_installation=True)

    def _verify_environment(self, attempt: _Attempt) -> None:
        if self._current_environment() != attempt.environment:
            raise EnvironmentMismatch("Dependency lock changed during the managed computation")

    def ensure_unchanged(self) -> None:
        attempt = self.attempt
        assert attempt is not None and self.project is not None
        self._audit.reading = True
        try:
            for path in self._source_paths().keys() | self.loaded.keys():
                self._current_source(path)
            self._verify_file_reads(attempt)
            self._verify_environment(attempt)
        finally:
            self._audit.reading = False
        if self._indeterminate or attempt.indeterminate_files:
            raise ValueError("Source files or read dependencies changed during the computation check")

    def _loaded_environment(self, environment: EnvironmentSnapshot) -> dict[str, str]:
        first_party = {"feedbax", *plugin_packages().values()}
        loaded = {
            canonicalize_name(distribution)
            for name in {module.split(".", 1)[0] for module in tuple(sys.modules)}
            for distribution in self._distribution_owners.get(name, ())
        }
        return {name: environment.versions[name] for name in sorted(loaded - first_party) if name in environment.versions}


@contextmanager
def numerical_work() -> Iterator[None]:
    """Suspend protected Host observation while executing covered numerical work."""
    if _recorder is None:
        yield
    else:
        with _recorder.numerical_work():
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


def set_execution_prefix(ref: ExecutionEvidenceRef) -> None:
    if _recorder is not None and (attempt := _recorder.attempt) is not None and ref not in attempt.prefixes:
        attempt.prefixes.append(ref)


def snapshot_execution(store: ImmutableArtifactBlobProvider) -> ReceiptEvidence | None:
    if _recorder is None or _recorder.attempt is None:
        return None
    return _recorder.snapshot(store)


def checkpoint_execution_endpoint(store: ImmutableArtifactBlobProvider, completed_updates: int) -> CheckpointEvidenceRoot | None:
    if _recorder is None or _recorder.attempt is None:
        return None
    return _recorder.checkpoint(store, completed_updates)


def assert_execution_unchanged() -> None:
    if _recorder is None or _recorder.attempt is None:
        raise RuntimeError("Validity checking requires a scoped Host recorder")
    _recorder.ensure_unchanged()


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
