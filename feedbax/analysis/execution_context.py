"""Explicit runtime-only bindings for portable staged execution descriptors."""

from __future__ import annotations

import os
import stat
from collections.abc import Collection, Mapping, Sequence
from dataclasses import dataclass, field, replace
from pathlib import Path, PurePosixPath
from types import MappingProxyType
from typing import Any
from urllib.parse import unquote, urlsplit

from pydantic import ValidationError

from feedbax.contracts.manifest import ParentRef
from feedbax.contracts.staged_execution import (
    STAGED_CHECKPOINT_CUSTODY_BACKEND,
    STAGED_EXECUTION_DESCRIPTOR_SCHEMA_ID,
    STAGED_EXECUTION_DESCRIPTOR_SCHEMA_VERSION,
    StagedExecutionDescriptor,
    validate_staged_binding_name,
)
from feedbax.persistence.artifact_custody import (
    ImmutableArtifactBlobProvider,
    open_immutable_artifact_blob_provider,
)
from feedbax.training.checkpoint_custody import (
    ResolvedCheckpointTransaction,
    resolve_checkpoint_custody_ref as resolve_bound_checkpoint_custody_ref,
)


class StagedExecutionContextError(ValueError):
    """Raised when a portable descriptor cannot be safely bound for execution."""


@dataclass(frozen=True, slots=True)
class StagedArtifactProviderRootBinding:
    """One explicit runtime root for a logical immutable artifact provider."""

    name: str
    root: Path | str


@dataclass(frozen=True, slots=True)
class StagedCheckpointCustodyRootBinding:
    """One explicit runtime root for a logical checkpoint-custody authority."""

    name: str
    root: Path | str


@dataclass(frozen=True, slots=True)
class StagedParentExecutionLocation:
    """A complete portable parent bound to its preflight-verified local location."""

    parent: ParentRef
    root: Path
    execution_uri: str


@dataclass(frozen=True, slots=True)
class StagedExecutionContext:
    """Validated runtime-only resources for registered staged recipes."""

    descriptor: StagedExecutionDescriptor | None
    opened_artifact_providers: Mapping[str, ImmutableArtifactBlobProvider]
    checkpoint_custody_roots: Mapping[str, Path]
    parent_execution_locations: tuple[StagedParentExecutionLocation, ...]
    _checkpoint_custody_root_identities: Mapping[str, tuple[int, int]] = field(
        default_factory=dict,
        repr=False,
        compare=False,
    )

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "opened_artifact_providers",
            MappingProxyType(dict(self.opened_artifact_providers)),
        )
        object.__setattr__(
            self,
            "checkpoint_custody_roots",
            MappingProxyType(dict(self.checkpoint_custody_roots)),
        )
        identities = dict(self._checkpoint_custody_root_identities)
        if not identities:
            identities = {
                name: _directory_identity(root, kind="checkpoint custody")
                for name, root in self.checkpoint_custody_roots.items()
            }
        if set(identities) != set(self.checkpoint_custody_roots):
            raise StagedExecutionContextError(
                "checkpoint custody root identities must exactly match bound roots"
            )
        object.__setattr__(
            self,
            "_checkpoint_custody_root_identities",
            MappingProxyType(identities),
        )
        object.__setattr__(
            self,
            "parent_execution_locations",
            tuple(self.parent_execution_locations),
        )

    def artifact_provider(self, name: str) -> ImmutableArtifactBlobProvider:
        """Return one explicitly bound immutable artifact provider by name."""
        validate_staged_binding_name(name)
        try:
            return self.opened_artifact_providers[name]
        except KeyError as exc:
            raise StagedExecutionContextError(
                f"staged artifact provider binding is unavailable: {name!r}"
            ) from exc

    def checkpoint_custody_root(self, name: str) -> Path:
        """Return one explicitly bound trusted checkpoint-custody root by name."""
        validate_staged_binding_name(name)
        try:
            root = self.checkpoint_custody_roots[name]
            identity = self._checkpoint_custody_root_identities[name]
        except KeyError as exc:
            raise StagedExecutionContextError(
                f"staged checkpoint custody binding is unavailable: {name!r}"
            ) from exc
        _require_directory_identity(root, identity, kind="checkpoint custody")
        return root

    def parent_execution_location(self, parent: ParentRef) -> StagedParentExecutionLocation:
        """Return the location whose complete ParentRef value equals ``parent``."""
        matches = [
            location for location in self.parent_execution_locations if location.parent == parent
        ]
        if not matches:
            raise StagedExecutionContextError(
                "staged parent execution location is unavailable for the complete ParentRef"
            )
        if len(matches) != 1:  # Construction prevents this; retain fail-closed lookup.
            raise StagedExecutionContextError(
                "staged parent execution location is ambiguous for the complete ParentRef"
            )
        return matches[0]

    def resolve_checkpoint_custody_ref(
        self,
        ref: ParentRef,
        binding_name: str | None = None,
        slot_names: Collection[str] | None = None,
    ) -> ResolvedCheckpointTransaction:
        """Resolve one checkpoint transaction under explicitly bound custody authority."""
        metadata_binding = ref.metadata.get("checkpoint_custody_binding")
        if metadata_binding is not None and (
            not isinstance(metadata_binding, str) or not metadata_binding
        ):
            raise StagedExecutionContextError(
                "ParentRef metadata.checkpoint_custody_binding must be a nonempty string"
            )
        if binding_name is not None:
            validate_staged_binding_name(binding_name)
        if (
            binding_name is not None
            and metadata_binding is not None
            and binding_name != metadata_binding
        ):
            raise StagedExecutionContextError(
                "explicit checkpoint custody binding disagrees with "
                "ParentRef metadata.checkpoint_custody_binding"
            )
        selected_binding = binding_name or metadata_binding
        if selected_binding is None:
            raise StagedExecutionContextError(
                "checkpoint custody resolution requires binding_name or "
                "ParentRef metadata.checkpoint_custody_binding"
            )
        validate_staged_binding_name(selected_binding)
        _validate_checkpoint_ref_uri(ref.uri)
        root = self.checkpoint_custody_root(selected_binding)
        expected_identity = self._checkpoint_custody_root_identities[selected_binding]
        try:
            return resolve_bound_checkpoint_custody_ref(
                ref,
                allowed_root=root,
                slot_names=slot_names,
            )
        finally:
            _require_directory_identity(
                root,
                expected_identity,
                kind="checkpoint custody",
            )


EMPTY_STAGED_EXECUTION_CONTEXT = StagedExecutionContext(
    descriptor=None,
    opened_artifact_providers={},
    checkpoint_custody_roots={},
    parent_execution_locations=(),
)


def resolve_staged_execution_context(
    descriptor: StagedExecutionDescriptor | Mapping[str, Any] | None,
    *,
    artifact_provider_bindings: Sequence[StagedArtifactProviderRootBinding] = (),
    checkpoint_custody_bindings: Sequence[StagedCheckpointCustodyRootBinding] = (),
) -> StagedExecutionContext:
    """Validate and bind every runtime resource before staged recipe effects."""
    if descriptor is None:
        if artifact_provider_bindings or checkpoint_custody_bindings:
            raise StagedExecutionContextError(
                "runtime staged bindings require an explicit StagedExecutionDescriptor"
            )
        return EMPTY_STAGED_EXECUTION_CONTEXT

    portable = _coerce_descriptor(descriptor)
    artifact_roots = _validated_binding_roots(
        artifact_provider_bindings,
        expected_names=set(portable.artifact_providers),
        kind="artifact provider",
    )
    checkpoint_roots = _validated_binding_roots(
        checkpoint_custody_bindings,
        expected_names=set(portable.checkpoint_custody),
        kind="checkpoint custody",
    )

    for name, checkpoint_spec in portable.checkpoint_custody.items():
        if checkpoint_spec.backend != STAGED_CHECKPOINT_CUSTODY_BACKEND:
            raise StagedExecutionContextError(
                f"unsupported checkpoint custody backend for {name!r}: {checkpoint_spec.backend!r}"
            )

    opened = {
        name: open_immutable_artifact_blob_provider(spec, explicit_root=artifact_roots[name])
        for name, spec in portable.artifact_providers.items()
    }
    return StagedExecutionContext(
        descriptor=portable,
        opened_artifact_providers=opened,
        checkpoint_custody_roots=checkpoint_roots,
        parent_execution_locations=(),
    )


def with_staged_parent_execution_locations(
    context: StagedExecutionContext,
    locations: Sequence[StagedParentExecutionLocation],
) -> StagedExecutionContext:
    """Return ``context`` with one-to-one preflight-verified parent locations."""
    normalized: list[StagedParentExecutionLocation] = []
    serialized_parents: set[str] = set()
    location_keys: set[tuple[Path, str]] = set()
    for location in locations:
        root = _validate_runtime_root(location.root, kind="parent execution")
        execution_uri = _validate_relative_execution_uri(location.execution_uri)
        parent_key = location.parent.model_dump_json(exclude_none=False)
        if parent_key in serialized_parents:
            raise StagedExecutionContextError(
                "staged execution context contains a duplicate complete ParentRef location"
            )
        location_key = (root, execution_uri)
        if location_key in location_keys:
            raise StagedExecutionContextError(
                "staged execution context contains a duplicate parent execution location"
            )
        serialized_parents.add(parent_key)
        location_keys.add(location_key)
        normalized.append(
            StagedParentExecutionLocation(
                parent=location.parent,
                root=root,
                execution_uri=execution_uri,
            )
        )
    return replace(context, parent_execution_locations=tuple(normalized))


def _coerce_descriptor(
    descriptor: StagedExecutionDescriptor | Mapping[str, Any],
) -> StagedExecutionDescriptor:
    if isinstance(descriptor, StagedExecutionDescriptor):
        descriptor = descriptor.model_dump(mode="python")
    if not isinstance(descriptor, Mapping):
        raise StagedExecutionContextError(
            "execution descriptor must be StagedExecutionDescriptor or a mapping"
        )
    missing = [field for field in ("schema_id", "schema_version") if field not in descriptor]
    if missing:
        raise StagedExecutionContextError(
            "StagedExecutionDescriptor requires explicit schema_id and schema_version; "
            f"missing {', '.join(missing)}"
        )
    if descriptor.get("schema_id") != STAGED_EXECUTION_DESCRIPTOR_SCHEMA_ID:
        raise StagedExecutionContextError(
            "unsupported StagedExecutionDescriptor schema_id; expected "
            f"{STAGED_EXECUTION_DESCRIPTOR_SCHEMA_ID!r}"
        )
    if descriptor.get("schema_version") != STAGED_EXECUTION_DESCRIPTOR_SCHEMA_VERSION:
        raise StagedExecutionContextError(
            "unsupported StagedExecutionDescriptor schema_version; current_version="
            f"{STAGED_EXECUTION_DESCRIPTOR_SCHEMA_VERSION!r}"
        )
    try:
        return StagedExecutionDescriptor.model_validate(descriptor)
    except ValidationError as exc:
        raise StagedExecutionContextError(f"invalid StagedExecutionDescriptor: {exc}") from exc


def _validated_binding_roots(
    bindings: Sequence[StagedArtifactProviderRootBinding]
    | Sequence[StagedCheckpointCustodyRootBinding],
    *,
    expected_names: set[str],
    kind: str,
) -> dict[str, Path]:
    raw_names = [binding.name for binding in bindings]
    for name in raw_names:
        try:
            validate_staged_binding_name(name)
        except ValueError as exc:
            raise StagedExecutionContextError(str(exc)) from exc
    if len(set(raw_names)) != len(raw_names):
        raise StagedExecutionContextError(f"duplicate staged {kind} runtime binding name")
    observed_names = set(raw_names)
    missing = sorted(expected_names - observed_names)
    extra = sorted(observed_names - expected_names)
    if missing or extra:
        details = []
        if missing:
            details.append(f"missing={missing}")
        if extra:
            details.append(f"extra={extra}")
        raise StagedExecutionContextError(
            f"staged {kind} binding names must exactly match the descriptor: " + ", ".join(details)
        )
    return {binding.name: _validate_runtime_root(binding.root, kind=kind) for binding in bindings}


def _validate_runtime_root(root: Path | str, *, kind: str) -> Path:
    if not isinstance(root, (Path, str)):
        raise StagedExecutionContextError(f"{kind} root must be an absolute path")
    raw = os.fspath(root)
    if not raw or "\0" in raw:
        raise StagedExecutionContextError(f"{kind} root must be nonempty and NUL-free")
    path = Path(raw)
    if not path.is_absolute():
        raise StagedExecutionContextError(
            f"{kind} root must be absolute; cwd, environment, and user expansion are forbidden"
        )
    if ".." in path.parts:
        raise StagedExecutionContextError(f"{kind} root must not contain lexical '..'")
    current = Path(path.anchor)
    try:
        for component in path.parts[1:]:
            current = current / component
            current_stat = current.lstat()
            if stat.S_ISLNK(current_stat.st_mode):
                raise StagedExecutionContextError(
                    f"{kind} root contains a symlink component: {current}"
                )
    except FileNotFoundError as exc:
        raise StagedExecutionContextError(f"{kind} root does not exist: {path}") from exc
    if not path.is_dir():
        raise StagedExecutionContextError(f"{kind} root is not a directory: {path}")
    return path.resolve(strict=True)


def _directory_identity(root: Path, *, kind: str) -> tuple[int, int]:
    flags = (
        os.O_RDONLY
        | getattr(os, "O_DIRECTORY", 0)
        | getattr(os, "O_NOFOLLOW", 0)
        | getattr(os, "O_CLOEXEC", 0)
    )
    if not getattr(os, "O_DIRECTORY", 0) or not getattr(os, "O_NOFOLLOW", 0):
        raise StagedExecutionContextError(
            f"{kind} authority requires no-follow directory descriptors"
        )
    try:
        descriptor = os.open(root, flags)
    except OSError as exc:
        raise StagedExecutionContextError(
            f"{kind} root is unavailable or was replaced: {root}"
        ) from exc
    try:
        root_stat = os.fstat(descriptor)
        if not stat.S_ISDIR(root_stat.st_mode):
            raise StagedExecutionContextError(f"{kind} root is not a directory: {root}")
        return root_stat.st_dev, root_stat.st_ino
    finally:
        os.close(descriptor)


def _require_directory_identity(
    root: Path,
    expected: tuple[int, int],
    *,
    kind: str,
) -> None:
    observed = _directory_identity(root, kind=kind)
    if observed != expected:
        raise StagedExecutionContextError(
            f"{kind} root authority was replaced after binding: {root}"
        )


def _validate_relative_execution_uri(uri: str) -> str:
    if not isinstance(uri, str) or not uri:
        raise StagedExecutionContextError("parent execution_uri must be a nonempty string")
    split = urlsplit(uri)
    if split.scheme or split.netloc or split.query or split.fragment:
        raise StagedExecutionContextError(
            f"parent execution_uri must be a plain relative path: {uri!r}"
        )
    decoded = unquote(split.path)
    if "\\" in decoded:
        raise StagedExecutionContextError(
            f"parent execution_uri contains an unsupported path separator: {uri!r}"
        )
    relative = PurePosixPath(decoded)
    if (
        relative.is_absolute()
        or not relative.parts
        or any(part in {"", ".", ".."} for part in relative.parts)
    ):
        raise StagedExecutionContextError(
            f"parent execution_uri escapes its explicit root: {uri!r}"
        )
    return relative.as_posix()


def _validate_checkpoint_ref_uri(uri: str | None) -> None:
    if uri is None or not uri:
        raise StagedExecutionContextError(
            "checkpoint custody ParentRef uri must be a nonempty relative path"
        )
    split = urlsplit(uri)
    if split.scheme or split.netloc or split.query or split.fragment:
        raise StagedExecutionContextError(
            "checkpoint custody ParentRef uri must be scheme-free, query-free, "
            "fragment-free, and relative"
        )
    decoded = unquote(split.path)
    if "\\" in decoded:
        raise StagedExecutionContextError(
            "checkpoint custody ParentRef uri contains an unsupported path separator"
        )
    relative = PurePosixPath(decoded)
    if (
        relative.is_absolute()
        or not relative.parts
        or any(part in {"", ".", ".."} for part in relative.parts)
    ):
        raise StagedExecutionContextError(
            "checkpoint custody ParentRef uri escapes its bound custody root"
        )


__all__ = [
    "EMPTY_STAGED_EXECUTION_CONTEXT",
    "StagedArtifactProviderRootBinding",
    "StagedCheckpointCustodyRootBinding",
    "StagedExecutionContext",
    "StagedExecutionContextError",
    "StagedParentExecutionLocation",
    "resolve_staged_execution_context",
    "with_staged_parent_execution_locations",
]
