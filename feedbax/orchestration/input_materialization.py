"""Driver-neutral materialization of authenticated orchestration inputs."""

from __future__ import annotations

import hashlib
import os
import stat
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path, PurePosixPath

from feedbax.contracts.manifest import ArtifactRef
from feedbax.contracts.staged_execution import validate_staged_binding_name
from feedbax.orchestration.bundle import ResolvedAssemblyInput, RunBundle
from feedbax.persistence.artifact_custody import ImmutableArtifactBlobProvider, open_immutable_artifact_blob_provider
from feedbax.training.checkpoint_custody import materialize_checkpoint_custody_archive


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
    return failures, observed


def preflight_input_provider_bindings(bundle: RunBundle, bindings: Sequence[InputProviderRootBinding]) -> tuple[list[str], list[dict[str, str]]]:
    failures, observed = preflight_resolved_inputs(bundle)
    try:
        providers = _bound_providers(bundle.resolved_inputs, bindings)
        for item in bundle.resolved_inputs:
            providers[item.custody.provider_binding].get_bytes(_artifact_ref(item))
    except Exception as exc:
        failures.append(str(exc))
    return failures, observed


def materialize_bundle_inputs(
    bundle: RunBundle, *, destination_root: Path | str,
    provider_bindings: Sequence[InputProviderRootBinding] = (),
) -> tuple[StagedInput, ...]:
    failures, _ = preflight_input_provider_bindings(bundle, provider_bindings)
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
    return tuple(staged)


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
