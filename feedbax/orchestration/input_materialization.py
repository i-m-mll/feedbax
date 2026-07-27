"""Driver-neutral materialization of authenticated orchestration inputs."""

from __future__ import annotations

import hashlib
import os
import shutil
import stat
import tempfile
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path, PurePosixPath

from feedbax.contracts.manifest import ArtifactRef
from feedbax.contracts.staged_execution import validate_staged_binding_name
from feedbax.orchestration.bundle import ResolvedAssemblyInput, RunBundle
from feedbax.orchestration.staged_root_custody import (
    MaterializedStagedRoot,
    StagedExecutionRootBindings,
    StagedRootCustody,
    StagedRootCustodyError,
    StagedRootSnapshotBinding,
    materialize_staged_root_snapshot,
    staged_execution_root_bindings,
    verify_staged_root_snapshot,
    verify_staged_root_snapshot_binding,
)
from feedbax.persistence.artifact_custody import (
    ImmutableArtifactBlobProvider,
    open_immutable_artifact_blob_provider,
)
from feedbax.training.checkpoint_custody import (
    materialize_checkpoint_custody_archive,
    publish_directory_no_replace,
)


class InputMaterializationError(ValueError):
    pass


@dataclass(frozen=True, slots=True)
class InputProviderRootBinding:
    name: str
    root: Path | str


@dataclass(frozen=True, slots=True)
class StagedInputFile:
    relative_path: str
    sha256: str
    size_bytes: int


@dataclass(frozen=True, slots=True)
class StagedInput:
    target_role: str
    destination: Path
    files: tuple[StagedInputFile, ...]


def preflight_resolved_inputs(bundle: RunBundle) -> tuple[list[str], list[dict[str, str]]]:
    failures: list[str] = []
    observed: list[dict[str, str]] = []
    roles: set[str] = set()
    for index, resolved in enumerate(bundle.resolved_inputs):
        source = resolved.custody
        try:
            validate_staged_binding_name(source.target_role)
        except ValueError as exc:
            failures.append(f"resolved_inputs[{index}].target_role: {exc}")
        if source.target_role in roles:
            failures.append(f"duplicate input target role: {source.target_role!r}")
        roles.add(source.target_role)
        if resolved.identity.digest.value != source.artifact.sha256:
            failures.append(f"resolved_inputs[{index}] identity/custody digest mismatch")
        parent = source.materializer.expected_parent_ref
        if not isinstance(parent.metadata.get("manifest_sha256"), str):
            failures.append(f"resolved_inputs[{index}] ParentRef lacks manifest digest")
        observed.append({"role": source.target_role, "provider_binding": source.provider_binding,
                         "artifact_sha256": source.artifact.sha256, "transaction_id": parent.id})
    for custody in bundle.staged_roots:
        observed.append(
            {
                "role": custody.binding_name,
                "provider_binding": custody.root_kind,
                "artifact_sha256": custody.content_sha256,
                "transaction_id": custody.custody_ref,
            }
        )
    return failures, observed


def preflight_input_provider_bindings(bundle: RunBundle, bindings: Sequence[InputProviderRootBinding]) -> tuple[list[str], list[dict[str, str]]]:
    failures, observed = preflight_resolved_inputs(bundle)
    observed = observed[: len(bundle.resolved_inputs)]
    try:
        providers = _bound_providers(bundle.resolved_inputs, bindings)
        for item in bundle.resolved_inputs:
            providers[item.custody.provider_binding].get_bytes(_artifact_ref(item))
    except Exception as exc:
        failures.append(str(exc))
    return failures, observed


def preflight_staged_root_bindings(
    bundle: RunBundle,
    bindings: Sequence[StagedRootSnapshotBinding],
) -> tuple[list[str], list[dict[str, str]]]:
    """Authenticate exact sealed staged-root bindings without external effects."""
    failures: list[str] = []
    roots: dict[tuple[str, str], StagedRootSnapshotBinding] = {}
    for binding in bindings:
        try:
            validate_staged_binding_name(binding.name)
        except ValueError as exc:
            failures.append(str(exc))
            continue
        key = (binding.kind, binding.name)
        if key in roots:
            failures.append(f"duplicate staged-root snapshot binding: {key!r}")
        roots[key] = binding
    expected = {
        (custody.root_kind, custody.binding_name): custody for custody in bundle.staged_roots
    }
    missing = sorted(set(expected) - set(roots))
    extra = sorted(set(roots) - set(expected))
    if missing or extra:
        failures.append(f"staged-root bindings differ; missing={missing!r} unexpected={extra!r}")
    observed: list[dict[str, str]] = []
    for key in sorted(set(expected) & set(roots)):
        custody = expected[key]
        try:
            root = verify_staged_root_snapshot_binding(roots[key])
            verify_staged_root_snapshot(custody, root)
        except (OSError, StagedRootCustodyError) as exc:
            failures.append(str(exc))
        observed.append(
            {
                "binding_name": custody.binding_name,
                "root_kind": custody.root_kind,
                "custody_ref": custody.custody_ref,
            }
        )
    return failures, observed


def preflight_bundle_input_bindings(
    bundle: RunBundle,
    *,
    provider_bindings: Sequence[InputProviderRootBinding],
    staged_root_bindings: Sequence[StagedRootSnapshotBinding],
) -> tuple[list[str], object]:
    """Preflight checkpoint archives and staged roots through one input boundary."""
    provider_failures, provider_observed = preflight_input_provider_bindings(
        bundle, provider_bindings
    )
    staged_failures, staged_observed = preflight_staged_root_bindings(bundle, staged_root_bindings)
    observed: object = provider_observed
    if bundle.staged_roots:
        observed = {
            "resolved_inputs": provider_observed,
            "staged_roots": staged_observed,
        }
    return [*provider_failures, *staged_failures], observed


def materialize_bundle_inputs(
    bundle: RunBundle, *, destination_root: Path | str,
    provider_bindings: Sequence[InputProviderRootBinding] = (),
    staged_root_bindings: Sequence[StagedRootSnapshotBinding] = (),
) -> tuple[StagedInput, ...]:
    failures, _ = preflight_bundle_input_bindings(
        bundle,
        provider_bindings=provider_bindings,
        staged_root_bindings=staged_root_bindings,
    )
    if failures:
        raise InputMaterializationError("; ".join(failures))
    providers = _bound_providers(bundle.resolved_inputs, provider_bindings)
    root = Path(destination_root).expanduser().resolve()
    (root / "inputs").mkdir(parents=True, exist_ok=True)
    staged: list[StagedInput] = []
    for resolved in bundle.resolved_inputs:
        source = resolved.custody
        destination = root / "inputs" / source.target_role
        provider = providers[source.provider_binding]
        try:
            if os.path.lexists(destination):
                raise InputMaterializationError(f"input destination already exists: {destination}")
            authority = source.materializer
            result = materialize_checkpoint_custody_archive(
                provider, _artifact_ref(resolved), destination,
                expected_parent_ref=authority.expected_parent_ref,
                expected_transaction_root_sha256=authority.expected_transaction_root_sha256,
            )
            expected = _expected_files(authority.expected_parent_ref.uri,
                                       result.resolved_transaction.manifest.slots)
            files = _file_manifest(root, destination, expected)
        except InputMaterializationError:
            raise
        except Exception as exc:
            raise InputMaterializationError(f"input {source.target_role!r} materialization failed: {exc}") from exc
        staged.append(StagedInput(source.target_role, destination, files))
    try:
        staged.extend(
            _materialize_staged_roots(
                bundle.staged_roots,
                root=root,
                bindings=staged_root_bindings,
            )
        )
    except StagedRootCustodyError as exc:
        raise InputMaterializationError(str(exc)) from exc
    return tuple(staged)


def staged_execution_bindings_for_bundle(
    bundle: RunBundle,
    *,
    inputs_root: Path | str,
) -> StagedExecutionRootBindings:
    """Authenticate materialized roots and project exact execution-context arguments."""
    root = Path(inputs_root).expanduser().resolve()
    materialized: list[MaterializedStagedRoot] = []
    for custody in bundle.staged_roots:
        destination = root / "staged-roots" / custody.root_kind / custody.binding_name
        verify_staged_root_snapshot(custody, destination)
        materialized.append(MaterializedStagedRoot(custody=custody, root=destination))
    return staged_execution_root_bindings(materialized)


def _materialize_staged_roots(
    custodies: Sequence[StagedRootCustody],
    *,
    root: Path,
    bindings: Sequence[StagedRootSnapshotBinding],
) -> tuple[StagedInput, ...]:
    if not custodies:
        return ()
    roots = {(binding.kind, binding.name): binding for binding in bindings}
    inputs_root = root / "inputs"
    final_root = inputs_root / "staged-roots"
    if os.path.lexists(final_root):
        raise StagedRootCustodyError(f"staged-root destination already exists: {final_root}")
    build_root = Path(tempfile.mkdtemp(prefix=".staged-roots-", dir=inputs_root))
    staged: list[StagedInput] = []
    try:
        for custody in custodies:
            key = (custody.root_kind, custody.binding_name)
            destination = build_root / custody.root_kind / custody.binding_name
            result = materialize_staged_root_snapshot(
                custody,
                verify_staged_root_snapshot_binding(roots[key]),
                destination,
            )
            files = tuple(
                StagedInputFile(
                    relative_path=(
                        Path("inputs")
                        / "staged-roots"
                        / custody.root_kind
                        / custody.binding_name
                        / record.relative_path
                    ).as_posix(),
                    sha256=record.sha256,
                    size_bytes=record.size_bytes,
                )
                for record in custody.files
            )
            staged.append(
                StagedInput(
                    target_role=(f"staged-roots/{custody.root_kind}/{custody.binding_name}"),
                    destination=(
                        inputs_root / "staged-roots" / custody.root_kind / custody.binding_name
                    ),
                    files=files,
                )
            )
            verify_staged_root_snapshot(custody, result.root)
            verify_staged_root_snapshot_binding(roots[key])
        parent_descriptor = os.open(
            inputs_root,
            os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | os.O_NOFOLLOW,
        )
        try:
            publish_directory_no_replace(
                parent_descriptor,
                build_root.name,
                final_root.name,
                expected_identity=os.stat(
                    build_root.name,
                    dir_fd=parent_descriptor,
                    follow_symlinks=False,
                ),
            )
        finally:
            os.close(parent_descriptor)
    except Exception:
        if build_root.exists():
            _remove_materialization_tree(build_root)
        raise
    return tuple(staged)


def _remove_materialization_tree(root: Path) -> None:
    for directory, subdirs, files in os.walk(root):
        directory_path = Path(directory)
        directory_path.chmod(0o700)
        for name in [*subdirs, *files]:
            path = directory_path / name
            if not path.is_symlink():
                path.chmod(0o700 if path.is_dir() else 0o600)
    shutil.rmtree(root)


def _bound_providers(inputs: Sequence[ResolvedAssemblyInput], bindings: Sequence[InputProviderRootBinding]) -> Mapping[str, ImmutableArtifactBlobProvider]:
    roots: dict[str, Path] = {}
    for binding in bindings:
        validate_staged_binding_name(binding.name)
        if binding.name in roots:
            raise InputMaterializationError(f"duplicate provider binding: {binding.name!r}")
        roots[binding.name] = Path(binding.root)
    specs = {item.custody.provider_binding: item.custody.provider for item in inputs}
    missing, extra = sorted(set(specs) - set(roots)), sorted(set(roots) - set(specs))
    if missing or extra:
        raise InputMaterializationError(f"input provider bindings differ; missing={missing!r} unexpected={extra!r}")
    return {name: open_immutable_artifact_blob_provider(spec, explicit_root=roots[name])
            for name, spec in specs.items()}


def _artifact_ref(resolved: ResolvedAssemblyInput) -> ArtifactRef:
    source, artifact = resolved.custody, resolved.custody.artifact
    return ArtifactRef(
        role="training_checkpoint_custody_archive",
        logical_name=f"{source.materializer.expected_parent_ref.id}.checkpoint-custody.tar.gz",
        artifact_id=artifact.artifact_id,
        sha256=artifact.sha256,
        media_type=artifact.media_type,
        size_bytes=artifact.size_bytes,
        storage_backend=artifact.storage_backend,
        uri=artifact.artifact_id,
    )


def _expected_files(uri: str | None, slots: Sequence[object]) -> set[str]:
    assert uri is not None
    parent = PurePosixPath(uri).parent
    return {"latest.json", uri, *(str(parent / slot.relative_path) for slot in slots)}


def _file_manifest(root: Path, destination: Path, expected: set[str]) -> tuple[StagedInputFile, ...]:
    records, actual = [], set()
    for path in sorted(destination.rglob("*")):
        mode = path.lstat().st_mode
        if stat.S_ISLNK(mode) or (not stat.S_ISDIR(mode) and not stat.S_ISREG(mode)):
            raise InputMaterializationError(f"unsupported materialized object: {path}")
        if path.is_file():
            data = path.read_bytes()
            actual.add(path.relative_to(destination).as_posix())
            records.append(StagedInputFile(path.relative_to(root).as_posix(),
                                           hashlib.sha256(data).hexdigest(), len(data)))
    if actual != expected:
        raise InputMaterializationError(f"materialized file set differs; missing={sorted(expected-actual)!r} "
                                        f"unexpected={sorted(actual-expected)!r}")
    return tuple(records)
