"""Explicit runtime-only bindings for portable staged execution descriptors."""

from __future__ import annotations

import copy
import hashlib
import os
import stat
import weakref
from collections.abc import Collection, Mapping, Sequence
from dataclasses import dataclass, field, replace
from pathlib import Path, PurePosixPath
from types import MappingProxyType
from typing import Any
from urllib.parse import unquote, urlsplit

import jax.tree as jt
import jax.tree_util as jtu
import numpy as np
from pydantic import BaseModel, ValidationError

from feedbax.analysis.manifest_inputs import (
    ResolvedManifestInput,
    is_authenticated_manifest_ref,
)
from feedbax.contracts.evaluation_states import (
    EVALUATION_STATES_ARTIFACT_ROLE,
    _treedef_structure_fingerprint,
    load_authenticated_evaluation_states_artifact,
)
from feedbax.contracts.manifest import (
    ArtifactRef,
    EvaluationRunManifest,
    ParentRef,
    load_manifest_bytes as parse_manifest_bytes,
    safe_manifest_key,
)
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
    _immutable_model_snapshot,
    _immutable_snapshot_value,
    resolve_checkpoint_custody_ref as resolve_bound_checkpoint_custody_ref,
)


class StagedExecutionContextError(ValueError):
    """Raised when a portable descriptor cannot be safely bound for execution."""


class StagedLocatorMissingError(StagedExecutionContextError):
    """Raised when a local evaluation-state artifact has no canonical locator."""


class StagedLocatorMismatchError(StagedExecutionContextError):
    """Raised when local evaluation-state artifact locators disagree."""


class StagedLocatorAbsoluteError(StagedExecutionContextError):
    """Raised when a local evaluation-state artifact claims an absolute URI."""


class StagedLocatorTraversalError(StagedExecutionContextError):
    """Raised when a local evaluation-state artifact locator escapes its root."""


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
class StagedManifestRootBinding:
    """One explicit runtime root for a retained Feedbax manifest store."""

    name: str
    root: Path | str


@dataclass(frozen=True, slots=True)
class StagedParentExecutionLocation:
    """A complete portable parent bound to its preflight-verified local location."""

    parent: ParentRef
    root: Path
    execution_uri: str
    artifact_provider: str | None = None


@dataclass(frozen=True, slots=True)
class StagedParentArtifactProviderBinding:
    """Map one authored provider label to one exact parent's runtime provider."""

    parent: ParentRef
    authored_provider: str
    runtime_provider: str


_FileState = tuple[int, int, int, int, int, int, int]


@dataclass(frozen=True, slots=True)
class _RetainedFileSnapshot:
    root: Path
    relative: str
    expected_root_identity: tuple[int, int]
    state: _FileState
    kind: str
    require_single_link: bool

    def is_current(self) -> bool:
        try:
            observed = _retained_file_state(
                self.root,
                self.relative,
                expected_root_identity=self.expected_root_identity,
                kind=self.kind,
                require_single_link=self.require_single_link,
            )
        except (OSError, StagedExecutionContextError):
            return False
        return observed == self.state


@dataclass(frozen=True, slots=True)
class _ManifestAuthorityKey:
    parent: str
    root: str
    execution_uri: str
    artifact_provider: str | None
    manifest_sha256: str
    manifest_size_bytes: int


@dataclass(frozen=True, slots=True)
class _EvaluationStatesAuthorityKey:
    manifest: _ManifestAuthorityKey
    states_sha256: str
    states_size_bytes: int
    requested_structure_fingerprint: str | None


@dataclass(frozen=True, slots=True)
class _CheckpointLookupKey:
    parent: str
    root: str
    binding_name: str
    slot_names: tuple[str, ...] | None


@dataclass(frozen=True, slots=True)
class _CheckpointAuthorityKey:
    lookup: _CheckpointLookupKey
    manifest_sha256: str
    manifest_size_bytes: int
    slots: tuple[tuple[str, str, int, str], ...]


@dataclass(frozen=True, slots=True)
class _MemoEntry:
    value: Any
    snapshots: tuple[_RetainedFileSnapshot, ...]

    def is_current(self) -> bool:
        return all(snapshot.is_current() for snapshot in self.snapshots)


@dataclass(frozen=True, slots=True)
class _ReconstructableEvaluationStates:
    """Verified immutable leaves for a fresh custom-PyTree wrapper per lookup."""

    structure: jtu.PyTreeDef
    leaves: tuple[Any, ...]

    def materialize(self) -> Any:
        structure = copy.deepcopy(self.structure)
        return structure.unflatten(list(self.leaves))


@dataclass(slots=True)
class _StagedExecutionMemo:
    manifests: dict[_ManifestAuthorityKey, _MemoEntry] = field(default_factory=dict)
    evaluation_states: dict[_EvaluationStatesAuthorityKey, _MemoEntry] = field(default_factory=dict)
    state_keys_by_object_id: dict[int, _EvaluationStatesAuthorityKey] = field(default_factory=dict)
    state_object_refs_by_id: dict[int, weakref.ReferenceType[Any]] = field(default_factory=dict)
    authenticated_channels: dict[_EvaluationStatesAuthorityKey, Any] = field(default_factory=dict)
    checkpoint_lookup: dict[_CheckpointLookupKey, _CheckpointAuthorityKey] = field(
        default_factory=dict
    )
    checkpoints: dict[_CheckpointAuthorityKey, _MemoEntry] = field(default_factory=dict)

    def invalidate_manifest(self, key: _ManifestAuthorityKey) -> None:
        self.manifests.pop(key, None)
        for states_key in tuple(self.evaluation_states):
            if states_key.manifest == key:
                self.invalidate_evaluation_states(states_key)

    def invalidate_evaluation_states(self, key: _EvaluationStatesAuthorityKey) -> None:
        self.evaluation_states.pop(key, None)
        for identity, candidate in tuple(self.state_keys_by_object_id.items()):
            if candidate == key:
                self.state_keys_by_object_id.pop(identity, None)
                self.state_object_refs_by_id.pop(identity, None)
        self.authenticated_channels.pop(key, None)

    def remember_evaluation_states(
        self,
        value: Any,
        key: _EvaluationStatesAuthorityKey,
        *,
        reconstructed: bool,
    ) -> None:
        identity = id(value)
        self.state_keys_by_object_id[identity] = key
        if not reconstructed:
            return
        try:
            reference = weakref.ref(
                value,
                lambda observed, identity=identity: self._forget_evaluation_states(
                    identity, observed
                ),
            )
        except TypeError:
            self.state_keys_by_object_id.pop(identity, None)
            return
        self.state_object_refs_by_id[identity] = reference

    def _forget_evaluation_states(
        self,
        identity: int,
        observed: weakref.ReferenceType[Any],
    ) -> None:
        if self.state_object_refs_by_id.get(identity) is observed:
            self.state_object_refs_by_id.pop(identity, None)
            self.state_keys_by_object_id.pop(identity, None)


@dataclass(frozen=True, slots=True)
class StagedExecutionContext:
    """Validated runtime-only resources for registered staged recipes."""

    descriptor: StagedExecutionDescriptor | None
    opened_artifact_providers: Mapping[str, ImmutableArtifactBlobProvider]
    checkpoint_custody_roots: Mapping[str, Path]
    parent_execution_locations: tuple[StagedParentExecutionLocation, ...]
    parent_artifact_provider_bindings: tuple[StagedParentArtifactProviderBinding, ...] = ()
    manifest_roots: Mapping[str, Path] = field(default_factory=dict)
    repo_root: Path | None = None
    _checkpoint_custody_root_identities: Mapping[str, tuple[int, int]] = field(
        default_factory=dict,
        repr=False,
        compare=False,
    )
    _artifact_provider_root_identities: Mapping[str, tuple[int, int]] = field(
        default_factory=dict,
        repr=False,
        compare=False,
    )
    _manifest_root_identities: Mapping[str, tuple[int, int]] = field(
        default_factory=dict,
        repr=False,
        compare=False,
    )
    _parent_execution_root_identities: tuple[tuple[int, int], ...] = field(
        default=(), repr=False, compare=False
    )
    _memo: _StagedExecutionMemo = field(
        default_factory=_StagedExecutionMemo,
        repr=False,
        compare=False,
    )

    def __post_init__(self) -> None:
        if self.repo_root is not None:
            object.__setattr__(self, "repo_root", Path(self.repo_root).resolve())
        object.__setattr__(
            self,
            "opened_artifact_providers",
            MappingProxyType(dict(self.opened_artifact_providers)),
        )
        artifact_identities = dict(self._artifact_provider_root_identities)
        if not artifact_identities:
            artifact_identities = {
                name: _directory_identity(Path(provider.root), kind="artifact provider")
                for name, provider in self.opened_artifact_providers.items()
            }
        if set(artifact_identities) != set(self.opened_artifact_providers):
            raise StagedExecutionContextError(
                "artifact provider root identities must exactly match opened providers"
            )
        object.__setattr__(
            self,
            "_artifact_provider_root_identities",
            MappingProxyType(artifact_identities),
        )
        object.__setattr__(
            self,
            "manifest_roots",
            MappingProxyType(dict(self.manifest_roots)),
        )
        manifest_identities = dict(self._manifest_root_identities)
        if not manifest_identities:
            manifest_identities = {
                name: _directory_identity(root, kind="manifest store")
                for name, root in self.manifest_roots.items()
            }
        if set(manifest_identities) != set(self.manifest_roots):
            raise StagedExecutionContextError(
                "manifest store root identities must exactly match bound roots"
            )
        object.__setattr__(
            self,
            "_manifest_root_identities",
            MappingProxyType(manifest_identities),
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
        bindings = tuple(self.parent_artifact_provider_bindings)
        binding_keys: set[tuple[str, str]] = set()
        for binding in bindings:
            validate_staged_binding_name(binding.authored_provider)
            validate_staged_binding_name(binding.runtime_provider)
            key = (
                binding.parent.model_dump_json(exclude_none=False),
                binding.authored_provider,
            )
            if key in binding_keys:
                raise StagedExecutionContextError(
                    "staged execution context contains a duplicate exact-parent "
                    "artifact-provider binding"
                )
            binding_keys.add(key)
        object.__setattr__(self, "parent_artifact_provider_bindings", bindings)
        parent_identities = tuple(self._parent_execution_root_identities)
        if not parent_identities:
            parent_identities = tuple(
                _directory_identity(location.root, kind="parent execution")
                for location in self.parent_execution_locations
            )
        if len(parent_identities) != len(self.parent_execution_locations):
            raise StagedExecutionContextError(
                "parent execution root identities must match parent locations"
            )
        object.__setattr__(self, "_parent_execution_root_identities", parent_identities)

    def artifact_provider(self, name: str) -> ImmutableArtifactBlobProvider:
        """Return one explicitly bound immutable artifact provider by name."""
        validate_staged_binding_name(name)
        try:
            provider = self.opened_artifact_providers[name]
        except KeyError as exc:
            raise StagedExecutionContextError(
                f"staged artifact provider binding is unavailable: {name!r}"
            ) from exc
        _require_directory_identity(
            Path(provider.root),
            self._artifact_provider_root_identities[name],
            kind="artifact provider",
        )
        return provider

    def artifact_provider_for_parent(
        self,
        parent: ParentRef,
        authored_provider: str,
    ) -> ImmutableArtifactBlobProvider:
        """Resolve one authored provider label under exact-parent runtime authority."""
        validate_staged_binding_name(authored_provider)
        matches = [
            binding
            for binding in self.parent_artifact_provider_bindings
            if binding.parent == parent and binding.authored_provider == authored_provider
        ]
        if not matches:
            raise StagedExecutionContextError(
                "staged artifact provider binding is unavailable for the exact parent "
                f"and authored provider {authored_provider!r}"
            )
        if len(matches) != 1:
            raise StagedExecutionContextError(
                "staged artifact provider binding is ambiguous for the exact parent "
                f"and authored provider {authored_provider!r}"
            )
        binding = matches[0]
        location = self.parent_execution_location(parent)
        if (
            location.artifact_provider is not None
            and location.artifact_provider != binding.runtime_provider
        ):
            raise StagedExecutionContextError(
                "exact-parent artifact provider binding disagrees with the parent "
                "execution provider"
            )
        provider = self.artifact_provider(binding.runtime_provider)
        if (
            location.artifact_provider is not None
            and Path(provider.root).resolve() != location.root.resolve()
        ):
            raise StagedExecutionContextError(
                "exact-parent artifact provider binding disagrees with the parent "
                "execution root"
            )
        return provider

    def resolve_manifest_input(self, parent: ParentRef) -> ResolvedManifestInput:
        """Resolve one authenticated manifest through its retained runtime authority."""
        location = self.parent_execution_location(parent)
        key = _manifest_authority_key(parent, location)
        cached = self._memo.manifests.get(key)
        if cached is not None:
            if cached.is_current():
                return cached.value
            self._memo.invalidate_manifest(key)
        if location.artifact_provider is None:
            location_index = self.parent_execution_locations.index(location)
            resolved, snapshot = _resolve_retained_local_manifest_input(
                parent,
                location,
                expected_root_identity=self._parent_execution_root_identities[location_index],
            )
            self._memo.manifests[key] = _MemoEntry(resolved, (snapshot,))
            return resolved
        if not is_authenticated_manifest_ref(parent):
            raise StagedExecutionContextError(
                "provider-backed manifest input requires an authenticated ParentRef"
            )
        provider = self.artifact_provider(location.artifact_provider)
        expected_root_identity = self._artifact_provider_root_identities[location.artifact_provider]
        digest = str(parent.metadata["manifest_sha256"])
        size_bytes = int(parent.metadata["size_bytes"])
        artifact_id = f"artifact://sha256/{digest}"
        try:
            relative = _provider_artifact_relative_path(digest)
            try:
                before = _retained_file_snapshot(
                    Path(provider.root),
                    relative,
                    expected_root_identity=expected_root_identity,
                    kind="provider-backed manifest",
                    require_single_link=True,
                )
            except StagedExecutionContextError:
                # Preserve the provider's established missing/integrity diagnostics.
                provider.get_bytes(artifact_id, size_bytes=size_bytes)
                raise
            raw_bytes = provider.get_bytes(artifact_id, size_bytes=size_bytes)
            if hashlib.sha256(raw_bytes).hexdigest() != digest:
                raise StagedExecutionContextError("provider-backed manifest digest changed")
            manifest = parse_manifest_bytes(raw_bytes)
            if manifest.kind != parent.kind or manifest.id != parent.id:
                raise StagedExecutionContextError(
                    "provider-backed manifest kind or id disagrees with ParentRef"
                )
            resolved = _immutable_resolved_manifest_input(
                ref=parent,
                manifest=manifest,
                path=Path(provider.root) / location.execution_uri,
                raw_bytes=raw_bytes,
            )
            snapshot = _retained_file_snapshot(
                Path(provider.root),
                relative,
                expected_root_identity=expected_root_identity,
                kind="provider-backed manifest",
                require_single_link=True,
            )
            if snapshot.state != before.state:
                raise StagedExecutionContextError(
                    "provider-backed manifest identity changed during read"
                )
            self._memo.manifests[key] = _MemoEntry(resolved, (snapshot,))
            return resolved
        finally:
            _require_directory_identity(
                Path(provider.root),
                expected_root_identity,
                kind="artifact provider",
            )

    def load_evaluation_states(
        self,
        parent: ParentRef,
        *,
        structure: jtu.PyTreeDef | None = None,
    ) -> Any:
        """Load v1/v2 states through the authority retained for an exact eval parent.

        Typed v3 staged prerequisites fail closed with
        ``EvaluationStatesCustodyUnavailable``.
        """
        if parent.kind != "EvaluationRunManifest" or parent.role != "evaluation_run":
            raise StagedExecutionContextError(
                "evaluation states require an EvaluationRunManifest evaluation_run parent"
            )
        location = self.parent_execution_location(parent)
        resolved = self.resolve_manifest_input(parent)
        manifest = resolved.manifest
        if not isinstance(manifest, EvaluationRunManifest):
            raise StagedExecutionContextError(
                "resolved prerequisite is not an EvaluationRunManifest"
            )
        if manifest.status != "completed":
            raise StagedExecutionContextError(
                "evaluation states require a completed EvaluationRunManifest"
            )
        artifacts = [
            artifact
            for artifact in manifest.artifacts
            if artifact.role == EVALUATION_STATES_ARTIFACT_ROLE
        ]
        if len(artifacts) != 1:
            raise StagedExecutionContextError(
                "completed EvaluationRunManifest must have exactly one evaluation_states artifact"
            )
        artifact = artifacts[0]
        key = _EvaluationStatesAuthorityKey(
            manifest=_manifest_authority_key(parent, location),
            states_sha256=str(artifact.sha256),
            states_size_bytes=int(artifact.size_bytes),
            requested_structure_fingerprint=(
                _treedef_structure_fingerprint(structure) if structure is not None else None
            ),
        )
        cached = self._memo.evaluation_states.get(key)
        if cached is not None:
            if cached.is_current():
                states = _materialize_evaluation_states(cached.value)
                self._memo.remember_evaluation_states(
                    states,
                    key,
                    reconstructed=isinstance(cached.value, _ReconstructableEvaluationStates),
                )
                return states
            self._memo.invalidate_evaluation_states(key)
        if location.artifact_provider is None:
            location_index = self.parent_execution_locations.index(location)
            expected_root_identity = self._parent_execution_root_identities[location_index]
            canonical = _require_canonical_local_artifact_locators(artifact)
            _require_directory_identity(
                location.root, expected_root_identity, kind="parent execution"
            )
            try:
                data, snapshot = _read_retained_local_artifact(
                    location.root,
                    canonical,
                    expected_root_identity=expected_root_identity,
                    expected_size=artifact.size_bytes,
                    expected_sha256=artifact.sha256,
                )
                states = load_authenticated_evaluation_states_artifact(
                    artifact,
                    manifest_id=manifest.id,
                    data=data,
                    structure=structure,
                )
                memo_value = _evaluation_states_memo_value(states)
                if memo_value is None:
                    return states
                self._memo.evaluation_states[key] = _MemoEntry(memo_value, (snapshot,))
                states = _materialize_evaluation_states(memo_value)
                self._memo.remember_evaluation_states(
                    states,
                    key,
                    reconstructed=isinstance(memo_value, _ReconstructableEvaluationStates),
                )
                return states
            finally:
                _require_directory_identity(
                    location.root, expected_root_identity, kind="parent execution"
                )

        provider = self.artifact_provider(location.artifact_provider)
        expected_root_identity = self._artifact_provider_root_identities[location.artifact_provider]
        try:
            relative = _provider_artifact_relative_path(str(artifact.sha256))
            try:
                before = _retained_file_snapshot(
                    Path(provider.root),
                    relative,
                    expected_root_identity=expected_root_identity,
                    kind="provider-backed evaluation_states artifact",
                    require_single_link=True,
                )
            except StagedExecutionContextError:
                # Preserve the provider's established missing/integrity diagnostics.
                provider.get_bytes(
                    f"artifact://sha256/{artifact.sha256}",
                    size_bytes=artifact.size_bytes,
                )
                raise
            data = provider.get_bytes(
                f"artifact://sha256/{artifact.sha256}", size_bytes=artifact.size_bytes
            )
            states = load_authenticated_evaluation_states_artifact(
                artifact,
                manifest_id=manifest.id,
                data=data,
                structure=structure,
            )
            memo_value = _evaluation_states_memo_value(states)
            snapshot = _retained_file_snapshot(
                Path(provider.root),
                relative,
                expected_root_identity=expected_root_identity,
                kind="provider-backed evaluation_states artifact",
                require_single_link=True,
            )
            if snapshot.state != before.state:
                raise StagedExecutionContextError(
                    "provider-backed evaluation_states artifact identity changed during read"
                )
            if memo_value is None:
                return states
            self._memo.evaluation_states[key] = _MemoEntry(memo_value, (snapshot,))
            states = _materialize_evaluation_states(memo_value)
            self._memo.remember_evaluation_states(
                states,
                key,
                reconstructed=isinstance(memo_value, _ReconstructableEvaluationStates),
            )
            return states
        finally:
            _require_directory_identity(
                Path(provider.root),
                expected_root_identity,
                kind="artifact provider",
            )

    def _cached_authenticated_channels(self, states: Any) -> Any | None:
        key = self._memo.state_keys_by_object_id.get(id(states))
        if key is None:
            return None
        entry = self._memo.evaluation_states.get(key)
        if entry is None or (
            not isinstance(entry.value, _ReconstructableEvaluationStates)
            and entry.value is not states
        ):
            return None
        return self._memo.authenticated_channels.get(key)

    def _cache_authenticated_channels(self, states: Any, channels: Any) -> None:
        key = self._memo.state_keys_by_object_id.get(id(states))
        if key is None:
            raise StagedExecutionContextError(
                "authenticated channels require a memoized evaluation-state snapshot"
            )
        entry = self._memo.evaluation_states.get(key)
        if entry is None or (
            not isinstance(entry.value, _ReconstructableEvaluationStates)
            and entry.value is not states
        ):
            raise StagedExecutionContextError(
                "authenticated channels disagree with the memoized state snapshot"
            )
        self._memo.authenticated_channels[key] = channels

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
        selection = _checkpoint_slot_selection_identity(slot_names)
        lookup_key = _CheckpointLookupKey(
            parent=ref.model_dump_json(exclude_none=False),
            root=os.fspath(root),
            binding_name=selected_binding,
            slot_names=selection,
        )
        authority_key = self._memo.checkpoint_lookup.get(lookup_key)
        if authority_key is not None:
            cached = self._memo.checkpoints.get(authority_key)
            if cached is not None and cached.is_current():
                return cached.value
            self._memo.checkpoint_lookup.pop(lookup_key, None)
            self._memo.checkpoints.pop(authority_key, None)
        try:
            try:
                manifest_before = _retained_file_snapshot(
                    root,
                    str(ref.uri),
                    expected_root_identity=expected_identity,
                    kind="checkpoint transaction manifest",
                    require_single_link=False,
                )
            except StagedExecutionContextError:
                # Preserve checkpoint custody's established public diagnostic.
                resolve_bound_checkpoint_custody_ref(
                    ref,
                    allowed_root=root,
                    slot_names=slot_names,
                )
                raise
            resolved = resolve_bound_checkpoint_custody_ref(
                ref,
                allowed_root=root,
                slot_names=slot_names,
            )
            resolved = replace(
                resolved,
                slots=_immutable_authenticated_snapshot(resolved.slots),
            )
            manifest_snapshot = _retained_file_snapshot(
                root,
                str(ref.uri),
                expected_root_identity=expected_identity,
                kind="checkpoint transaction manifest",
                require_single_link=False,
            )
            if manifest_snapshot.state != manifest_before.state:
                raise StagedExecutionContextError(
                    "checkpoint transaction manifest identity changed during resolution"
                )
            slots_by_name = {slot.slot: slot for slot in resolved.manifest.slots}
            selected_slots = tuple(slots_by_name[name] for name in resolved.slots)
            slot_snapshots = tuple(
                _retained_file_snapshot(
                    root,
                    _checkpoint_slot_relative_path(str(ref.uri), slot.relative_path),
                    expected_root_identity=expected_identity,
                    kind=f"checkpoint slot {slot.slot!r}",
                    require_single_link=False,
                )
                for slot in selected_slots
            )
            authority_key = _CheckpointAuthorityKey(
                lookup=lookup_key,
                manifest_sha256=resolved.manifest_sha256,
                manifest_size_bytes=manifest_snapshot.state[4],
                slots=tuple(
                    (slot.slot, slot.sha256, slot.size_bytes, slot.relative_path)
                    for slot in selected_slots
                ),
            )
            self._memo.checkpoint_lookup[lookup_key] = authority_key
            self._memo.checkpoints[authority_key] = _MemoEntry(
                resolved,
                (manifest_snapshot, *slot_snapshots),
            )
            return resolved
        finally:
            _require_directory_identity(
                root,
                expected_identity,
                kind="checkpoint custody",
            )


EMPTY_STAGED_EXECUTION_CONTEXT = StagedExecutionContext(
    descriptor=None,
    opened_artifact_providers={},
    manifest_roots={},
    checkpoint_custody_roots={},
    parent_execution_locations=(),
)


def resolve_staged_execution_context(
    descriptor: StagedExecutionDescriptor | Mapping[str, Any] | None,
    *,
    artifact_provider_bindings: Sequence[StagedArtifactProviderRootBinding] = (),
    manifest_root_bindings: Sequence[StagedManifestRootBinding] = (),
    checkpoint_custody_bindings: Sequence[StagedCheckpointCustodyRootBinding] = (),
) -> StagedExecutionContext:
    """Validate and bind every runtime resource before staged recipe effects."""
    if descriptor is None:
        if artifact_provider_bindings or manifest_root_bindings or checkpoint_custody_bindings:
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
    manifest_roots = _validated_binding_roots(
        manifest_root_bindings,
        expected_names=None,
        kind="manifest store",
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
        manifest_roots=manifest_roots,
        checkpoint_custody_roots=checkpoint_roots,
        parent_execution_locations=(),
    )


def with_staged_repo_root(
    context: StagedExecutionContext,
    repo_root: Path | str | None,
) -> StagedExecutionContext:
    """Bind one explicit repository root without changing durable execution identity."""
    if repo_root is None:
        return context
    resolved = Path(repo_root).resolve()
    if context.repo_root is not None:
        if context.repo_root != resolved:
            raise StagedExecutionContextError(
                "execution_context repo_root disagrees with the explicit repo_root argument"
            )
        return context
    return replace(context, repo_root=resolved)


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
                artifact_provider=location.artifact_provider,
            )
        )
    return replace(
        context,
        parent_execution_locations=tuple(normalized),
        parent_artifact_provider_bindings=(),
        _parent_execution_root_identities=(),
        _memo=_StagedExecutionMemo(),
    )


def with_staged_parent_artifact_provider_bindings(
    context: StagedExecutionContext,
    bindings: Sequence[StagedParentArtifactProviderBinding],
) -> StagedExecutionContext:
    """Bind authored provider labels to preflight-verified exact parent locations."""
    bound = replace(
        context,
        parent_artifact_provider_bindings=tuple(bindings),
        _memo=_StagedExecutionMemo(),
    )
    for binding in bound.parent_artifact_provider_bindings:
        bound.artifact_provider_for_parent(binding.parent, binding.authored_provider)
    return bound


def with_staged_manifest_provider_inputs(
    context: StagedExecutionContext,
    parents: Sequence[ParentRef],
) -> StagedExecutionContext:
    """Bind authenticated parents found in exactly one explicit retained authority.

    Retained manifest-store and artifact-provider roots remain runtime-only
    logical bindings. This lookup never materializes manifest bytes under the
    analysis output root.
    """
    locations = list(context.parent_execution_locations)
    located_parents = {
        location.parent.model_dump_json(exclude_none=False) for location in locations
    }
    for parent in parents:
        serialized = parent.model_dump_json(exclude_none=False)
        if serialized in located_parents:
            continue
        if not is_authenticated_manifest_ref(parent):
            raise StagedExecutionContextError(
                "manifest-provider lookup requires an authenticated ParentRef"
            )
        digest = str(parent.metadata["manifest_sha256"])
        size_bytes = int(parent.metadata["size_bytes"])
        artifact_id = f"artifact://sha256/{digest}"
        matches: list[tuple[str, Path, str, str | None]] = []
        if parent.kind == "TrainingRunManifest":
            relative = f"manifests/training_runs/{safe_manifest_key(parent.id)}.json"
            for name, root in context.manifest_roots.items():
                _require_directory_identity(
                    root,
                    context._manifest_root_identities[name],
                    kind="manifest store",
                )
                if (root / relative).is_file():
                    matches.append((name, root, relative, None))
        for name, provider in context.opened_artifact_providers.items():
            try:
                relative = provider.canonical_relative_path(
                    artifact_id,
                    size_bytes=size_bytes,
                )
            except FileNotFoundError:
                continue
            matches.append((name, Path(provider.root), relative.as_posix(), name))
        if not matches:
            raise StagedExecutionContextError(
                f"authenticated manifest {parent.id!r} is unavailable in the bound "
                "manifest stores and artifact providers"
            )
        if len(matches) > 1:
            names = ", ".join(repr(name) for name, _root, _relative, _provider in matches)
            raise StagedExecutionContextError(
                f"authenticated manifest {parent.id!r} is duplicated across authorities: {names}"
            )
        _name, root, execution_uri, artifact_provider = matches[0]
        locations.append(
            StagedParentExecutionLocation(
                parent=parent,
                root=root,
                execution_uri=execution_uri,
                artifact_provider=artifact_provider,
            )
        )
        located_parents.add(serialized)
    bound = with_staged_parent_execution_locations(context, locations)
    provider_bindings = list(context.parent_artifact_provider_bindings)
    binding_keys = {
        (
            binding.parent.model_dump_json(exclude_none=False),
            binding.authored_provider,
        )
        for binding in provider_bindings
    }
    for location in bound.parent_execution_locations:
        if location.artifact_provider is None:
            continue
        key = (
            location.parent.model_dump_json(exclude_none=False),
            location.artifact_provider,
        )
        if key not in binding_keys:
            provider_bindings.append(
                StagedParentArtifactProviderBinding(
                    parent=location.parent,
                    authored_provider=location.artifact_provider,
                    runtime_provider=location.artifact_provider,
                )
            )
            binding_keys.add(key)
    bound = with_staged_parent_artifact_provider_bindings(bound, provider_bindings)
    for parent in parents:
        bound.resolve_manifest_input(parent)
    return bound


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
    | Sequence[StagedManifestRootBinding]
    | Sequence[StagedCheckpointCustodyRootBinding],
    *,
    expected_names: set[str] | None,
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
    if expected_names is not None:
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
                f"staged {kind} binding names must exactly match the descriptor: "
                + ", ".join(details)
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
    descriptors = _open_directory_chain_no_follow(path, kind=kind)
    for descriptor in reversed(descriptors):
        os.close(descriptor)
    return path


def _directory_identity(root: Path, *, kind: str) -> tuple[int, int]:
    descriptors = _open_directory_chain_no_follow(root, kind=kind)
    try:
        root_stat = os.fstat(descriptors[-1])
        return root_stat.st_dev, root_stat.st_ino
    finally:
        for descriptor in reversed(descriptors):
            os.close(descriptor)


def _open_directory_chain_no_follow(root: Path, *, kind: str) -> list[int]:
    """Open every component of an absolute directory path without following links."""
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
    if not root.is_absolute():
        raise StagedExecutionContextError(f"{kind} root must be absolute: {root}")
    descriptors: list[int] = []
    try:
        current = os.open(root.anchor, flags)
        descriptors.append(current)
        for component in root.parts[1:]:
            current = os.open(component, flags, dir_fd=current)
            descriptors.append(current)
    except OSError as exc:
        for descriptor in reversed(descriptors):
            os.close(descriptor)
        raise StagedExecutionContextError(
            f"{kind} root is unavailable, unsafe, or was replaced: {root}"
        ) from exc
    return descriptors


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


def _file_state(file_stat: os.stat_result) -> _FileState:
    return (
        file_stat.st_dev,
        file_stat.st_ino,
        file_stat.st_mode,
        file_stat.st_nlink,
        file_stat.st_size,
        file_stat.st_mtime_ns,
        file_stat.st_ctime_ns,
    )


def _retained_file_state(
    root: Path,
    relative: str,
    *,
    expected_root_identity: tuple[int, int],
    kind: str,
    require_single_link: bool,
) -> _FileState:
    """Return one cheap no-follow file identity under a retained root."""
    relative = _validate_relative_execution_uri(relative)
    parts = PurePosixPath(relative).parts
    descriptors = _open_directory_chain_no_follow(root, kind=kind)
    directory_flags = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | getattr(os, "O_CLOEXEC", 0)
    try:
        current = descriptors[-1]
        root_stat = os.fstat(current)
        if (root_stat.st_dev, root_stat.st_ino) != expected_root_identity:
            raise StagedExecutionContextError(f"{kind} root authority changed")
        for component in parts[:-1]:
            current = os.open(component, directory_flags, dir_fd=current)
            descriptors.append(current)
        observed = os.stat(parts[-1], dir_fd=current, follow_symlinks=False)
        if not stat.S_ISREG(observed.st_mode):
            raise StagedExecutionContextError(f"{kind} is not a regular file")
        if require_single_link and observed.st_nlink != 1:
            raise StagedExecutionContextError(f"{kind} has mutable hard-link aliases")
        return _file_state(observed)
    except OSError as exc:
        raise StagedExecutionContextError(
            f"{kind} path traverses a symlink or unsafe component"
        ) from exc
    finally:
        for descriptor in reversed(descriptors):
            os.close(descriptor)


def _retained_file_snapshot(
    root: Path,
    relative: str,
    *,
    expected_root_identity: tuple[int, int],
    kind: str,
    require_single_link: bool,
) -> _RetainedFileSnapshot:
    relative = _validate_relative_execution_uri(relative)
    return _RetainedFileSnapshot(
        root=root,
        relative=relative,
        expected_root_identity=expected_root_identity,
        state=_retained_file_state(
            root,
            relative,
            expected_root_identity=expected_root_identity,
            kind=kind,
            require_single_link=require_single_link,
        ),
        kind=kind,
        require_single_link=require_single_link,
    )


def _manifest_authority_key(
    parent: ParentRef,
    location: StagedParentExecutionLocation,
) -> _ManifestAuthorityKey:
    if not is_authenticated_manifest_ref(parent):
        raise StagedExecutionContextError(
            "staged manifest input requires an authenticated ParentRef"
        )
    return _ManifestAuthorityKey(
        parent=parent.model_dump_json(exclude_none=False),
        root=os.fspath(location.root),
        execution_uri=location.execution_uri,
        artifact_provider=location.artifact_provider,
        manifest_sha256=str(parent.metadata["manifest_sha256"]),
        manifest_size_bytes=int(parent.metadata["size_bytes"]),
    )


def _provider_artifact_relative_path(digest: str) -> str:
    return f"artifacts/sha256/{digest[:2]}/{digest}"


def _checkpoint_slot_selection_identity(
    slot_names: Collection[str] | None,
) -> tuple[str, ...] | None:
    if slot_names is None:
        return None
    if isinstance(slot_names, (str, bytes, bytearray)):
        return (str(slot_names),)
    requested = tuple(slot_names)
    if (
        requested
        and all(isinstance(name, str) and name for name in requested)
        and len(requested) == len(set(requested))
    ):
        return tuple(sorted(requested))
    return requested


def _checkpoint_slot_relative_path(manifest_relative: str, slot_relative: str) -> str:
    combined = os.path.normpath(
        (PurePosixPath(manifest_relative).parent / PurePosixPath(slot_relative)).as_posix()
    )
    return _validate_relative_execution_uri(combined)


def _mark_numpy_arrays_read_only(value: Any, *, _seen: set[int] | None = None) -> None:
    """Freeze NumPy arrays below real containers, including mapping proxies."""
    if isinstance(value, np.ndarray):
        value.flags.writeable = False
        return
    if isinstance(value, (str, bytes, bytearray, memoryview)):
        return
    seen = set() if _seen is None else _seen
    identity = id(value)
    if identity in seen:
        return
    seen.add(identity)
    if isinstance(value, BaseModel):
        for field_name in type(value).model_fields:
            _mark_numpy_arrays_read_only(getattr(value, field_name), _seen=seen)
        return
    if isinstance(value, Mapping):
        for key, item in value.items():
            _mark_numpy_arrays_read_only(key, _seen=seen)
            _mark_numpy_arrays_read_only(item, _seen=seen)
        return
    if isinstance(value, Sequence):
        for item in value:
            _mark_numpy_arrays_read_only(item, _seen=seen)
        return
    try:
        leaves = jt.leaves(value)
    except (TypeError, ValueError):
        return
    if len(leaves) == 1 and leaves[0] is value:
        return
    for leaf in leaves:
        _mark_numpy_arrays_read_only(leaf, _seen=seen)


def _immutable_authenticated_snapshot(value: Any) -> Any:
    """Return a recursively immutable snapshot of already-verified values."""
    _mark_numpy_arrays_read_only(value)
    return _immutable_snapshot_value(value)


def _evaluation_states_memo_value(value: Any) -> Any | None:
    """Freeze ordinary states or retain safe leaves for custom-PyTree rebuilds."""
    _mark_numpy_arrays_read_only(value)
    structure = jt.structure(value)
    if not _structure_has_custom_nodes(structure):
        return _immutable_snapshot_value(value)
    frozen_leaves: list[Any] = []
    for leaf in jt.leaves(value):
        frozen = _immutable_snapshot_value(leaf)
        if frozen is leaf and not _is_safe_shared_leaf(leaf):
            return None
        frozen_leaves.append(frozen)
    return _ReconstructableEvaluationStates(
        structure=copy.deepcopy(structure),
        leaves=tuple(frozen_leaves),
    )


def _materialize_evaluation_states(value: Any) -> Any:
    if isinstance(value, _ReconstructableEvaluationStates):
        return value.materialize()
    return value


def _structure_has_custom_nodes(structure: jtu.PyTreeDef) -> bool:
    node_data = structure.node_data()
    if node_data is not None:
        node_type, _aux = node_data
        if node_type not in {dict, list, tuple, type(None)} and not (
            isinstance(node_type, type) and issubclass(node_type, tuple)
        ):
            return True
    return any(_structure_has_custom_nodes(child) for child in structure.children())


def _is_safe_shared_leaf(value: Any) -> bool:
    if isinstance(value, np.ndarray):
        return not value.flags.writeable
    return value is None or isinstance(
        value,
        (bool, int, float, complex, str, bytes, np.generic),
    )


def _immutable_resolved_manifest_input(
    *,
    ref: ParentRef,
    manifest: Any,
    path: Path,
    raw_bytes: bytes,
) -> ResolvedManifestInput:
    """Freeze the parsed authority while retaining its exact authenticated bytes."""
    return ResolvedManifestInput(
        ref=_immutable_model_snapshot(ref),
        manifest=_immutable_model_snapshot(manifest),
        path=path,
        raw_bytes=raw_bytes,
    )


def _resolve_retained_local_manifest_input(
    parent: ParentRef,
    location: StagedParentExecutionLocation,
    *,
    expected_root_identity: tuple[int, int],
) -> tuple[ResolvedManifestInput, _RetainedFileSnapshot]:
    """Resolve an authenticated manifest through the exact retained root authority."""
    if not is_authenticated_manifest_ref(parent):
        raise StagedExecutionContextError(
            "local evaluation manifest requires an authenticated ParentRef"
        )
    execution_uri = _validate_relative_execution_uri(location.execution_uri)
    digest = str(parent.metadata["manifest_sha256"])
    size_bytes = int(parent.metadata["size_bytes"])
    raw_bytes, snapshot = _read_retained_local_file(
        location.root,
        execution_uri,
        expected_root_identity=expected_root_identity,
        expected_size=size_bytes,
        expected_sha256=digest,
        kind="local evaluation manifest",
        require_single_link=True,
    )
    manifest = parse_manifest_bytes(raw_bytes)
    if manifest.kind != parent.kind or manifest.id != parent.id:
        raise StagedExecutionContextError(
            "local evaluation manifest kind or id disagrees with ParentRef"
        )
    return (
        _immutable_resolved_manifest_input(
            ref=parent,
            manifest=manifest,
            path=location.root / execution_uri,
            raw_bytes=raw_bytes,
        ),
        snapshot,
    )


def _require_canonical_local_artifact_locators(
    artifact: ArtifactRef,
) -> str:
    """Reject ambiguous or noncanonical local artifact locator claims."""
    locators = {
        "uri": artifact.uri,
        "metadata.relative_path": artifact.metadata.get("relative_path"),
    }
    present = {name: value for name, value in locators.items() if value is not None}
    if not present:
        raise StagedLocatorMissingError(
            "local evaluation_states artifact requires a canonical relative locator"
        )
    uri = artifact.uri
    if isinstance(uri, str) and Path(uri).is_absolute():
        raise StagedLocatorAbsoluteError(
            "local evaluation_states uri is a machine-local absolute path; "
            f"a canonical relative locator is required: {uri}"
        )
    for name, value in present.items():
        try:
            _validate_relative_execution_uri(value)
        except (StagedExecutionContextError, TypeError) as exc:
            raise StagedLocatorTraversalError(
                f"local evaluation_states locator escapes its explicit root: {value}"
            ) from exc
    canonical = artifact.metadata.get("relative_path")
    if not isinstance(canonical, str):
        raise StagedLocatorMissingError(
            "local evaluation_states artifact requires a canonical relative locator"
        )
    canonical = _validate_relative_execution_uri(canonical)
    for name, value in present.items():
        if value != canonical:
            raise StagedLocatorMismatchError(
                f"local evaluation_states {name} must equal canonical path {canonical!r}"
            )
    return canonical


def _read_retained_local_artifact(
    root: Path,
    relative: str,
    *,
    expected_root_identity: tuple[int, int],
    expected_size: int,
    expected_sha256: str,
) -> tuple[bytes, _RetainedFileSnapshot]:
    """Read one root-relative regular single-link file without following aliases."""
    return _read_retained_local_file(
        root,
        relative,
        expected_root_identity=expected_root_identity,
        expected_size=expected_size,
        expected_sha256=expected_sha256,
        kind="local evaluation_states artifact",
        require_single_link=True,
    )


def _read_retained_local_file(
    root: Path,
    relative: str,
    *,
    expected_root_identity: tuple[int, int],
    expected_size: int,
    expected_sha256: str,
    kind: str,
    require_single_link: bool,
) -> tuple[bytes, _RetainedFileSnapshot]:
    """Read one authenticated file relative to an exact retained root authority."""
    parts = PurePosixPath(relative).parts
    directory_flags = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | getattr(os, "O_CLOEXEC", 0)
    file_flags = os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK | getattr(os, "O_CLOEXEC", 0)
    descriptors: list[int] = []
    directory_records: list[tuple[int, str, tuple[int, int]]] = []
    try:
        descriptors.extend(_open_directory_chain_no_follow(root, kind=kind))
        current = descriptors[-1]
        root_identity = os.fstat(current)
        if (root_identity.st_dev, root_identity.st_ino) != expected_root_identity:
            raise StagedExecutionContextError(f"{kind} root authority changed before read")
        for component in parts[:-1]:
            parent_descriptor = current
            current = os.open(component, directory_flags, dir_fd=parent_descriptor)
            descriptors.append(current)
            component_stat = os.fstat(current)
            directory_records.append(
                (
                    parent_descriptor,
                    component,
                    (component_stat.st_dev, component_stat.st_ino),
                )
            )
        path_before = os.stat(parts[-1], dir_fd=current, follow_symlinks=False)
        if stat.S_ISLNK(path_before.st_mode):
            raise StagedExecutionContextError(f"{kind} must not be a symlink")
        file_descriptor = os.open(parts[-1], file_flags, dir_fd=current)
        descriptors.append(file_descriptor)
        before = os.fstat(file_descriptor)
        if not stat.S_ISREG(before.st_mode):
            raise StagedExecutionContextError(f"{kind} is not regular")
        if require_single_link and before.st_nlink != 1:
            raise StagedExecutionContextError(f"{kind} has mutable hard-link aliases")
        if (before.st_dev, before.st_ino) != (path_before.st_dev, path_before.st_ino):
            raise StagedExecutionContextError(f"{kind} identity changed before read")
        if before.st_size != expected_size:
            raise StagedExecutionContextError(f"{kind} size mismatch")
        chunks: list[bytes] = []
        while chunk := os.read(file_descriptor, 1024 * 1024):
            chunks.append(chunk)
        after = os.fstat(file_descriptor)
        if require_single_link and after.st_nlink != 1:
            raise StagedExecutionContextError(f"{kind} hard-link count changed during read")
        if (
            before.st_dev,
            before.st_ino,
            before.st_size,
            before.st_mtime_ns,
            before.st_ctime_ns,
        ) != (
            after.st_dev,
            after.st_ino,
            after.st_size,
            after.st_mtime_ns,
            after.st_ctime_ns,
        ):
            raise StagedExecutionContextError(f"{kind} identity changed during read")
        path_after = os.stat(parts[-1], dir_fd=current, follow_symlinks=False)
        if (
            path_before.st_dev,
            path_before.st_ino,
            path_before.st_mode,
            path_before.st_nlink,
            path_before.st_size,
            path_before.st_mtime_ns,
            path_before.st_ctime_ns,
        ) != (
            path_after.st_dev,
            path_after.st_ino,
            path_after.st_mode,
            path_after.st_nlink,
            path_after.st_size,
            path_after.st_mtime_ns,
            path_after.st_ctime_ns,
        ):
            raise StagedExecutionContextError(f"{kind} path changed during read")
        for parent_descriptor, component, expected_identity in directory_records:
            component_after = os.stat(
                component,
                dir_fd=parent_descriptor,
                follow_symlinks=False,
            )
            if (
                not stat.S_ISDIR(component_after.st_mode)
                or (component_after.st_dev, component_after.st_ino) != expected_identity
            ):
                raise StagedExecutionContextError(f"{kind} directory identity changed during read")
        _require_directory_identity(root, expected_root_identity, kind=kind)
        data = b"".join(chunks)
        if hashlib.sha256(data).hexdigest() != expected_sha256:
            raise StagedExecutionContextError(f"{kind} sha256 mismatch")
        return (
            data,
            _RetainedFileSnapshot(
                root=root,
                relative=relative,
                expected_root_identity=expected_root_identity,
                state=_file_state(path_after),
                kind=kind,
                require_single_link=require_single_link,
            ),
        )
    except OSError as exc:
        raise StagedExecutionContextError(
            f"{kind} path traverses a symlink or unsafe component"
        ) from exc
    finally:
        for descriptor in reversed(descriptors):
            os.close(descriptor)


def _validate_relative_execution_uri(uri: str) -> str:
    if not isinstance(uri, str) or not uri:
        raise StagedExecutionContextError("parent execution_uri must be a nonempty string")
    split = urlsplit(uri)
    if split.scheme or split.netloc or split.query or split.fragment:
        raise StagedExecutionContextError(
            f"parent execution_uri must be a plain relative path: {uri!r}"
        )
    decoded = unquote(split.path)
    if "\x00" in decoded or "\\" in decoded:
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
    if not isinstance(uri, str) or not uri:
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
    if "\x00" in decoded or "\\" in decoded:
        raise StagedExecutionContextError(
            "checkpoint custody ParentRef uri contains an unsupported path separator"
        )
    raw_parts = decoded.split("/")
    relative = PurePosixPath(decoded)
    if relative.is_absolute():
        raise StagedExecutionContextError(
            "checkpoint custody ParentRef uri must be relative to its bound custody root"
        )
    if ".." in raw_parts:
        raise StagedExecutionContextError(
            "checkpoint custody ParentRef uri escapes its bound custody root"
        )
    if not relative.parts or any(part in {"", "."} for part in raw_parts):
        raise StagedExecutionContextError(
            "checkpoint custody ParentRef uri must not contain empty or dot path segments"
        )


__all__ = [
    "EMPTY_STAGED_EXECUTION_CONTEXT",
    "StagedArtifactProviderRootBinding",
    "StagedCheckpointCustodyRootBinding",
    "StagedExecutionContext",
    "StagedExecutionContextError",
    "StagedManifestRootBinding",
    "StagedParentExecutionLocation",
    "resolve_staged_execution_context",
    "with_staged_manifest_provider_inputs",
    "with_staged_parent_execution_locations",
    "with_staged_repo_root",
]
