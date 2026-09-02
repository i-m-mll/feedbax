"""Materialize Feedbax model artifacts into executable PyTree templates."""

from __future__ import annotations

import re
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import equinox as eqx
import jax.numpy as jnp
import jax.tree_util as jtu

from feedbax.contracts.artifact_schema import (
    ARRAY_STORE_SCHEMA_VERSION,
    ArrayStore,
    ArtifactSchemaError,
    ArrayStoreValidationError,
    NPZ_BACKEND_VERSION,
    read_npz_array_store,
    validate_role_address,
)
from feedbax.contracts.base import sha256_file
from feedbax.contracts.manifest import (
    SCHEMA_VERSION,
    ModelArtifactManifest,
    load_manifest,
)


class UnsupportedArtifactSchemaError(ArtifactSchemaError):
    """Raised when an artifact must be migrated before materialization."""


def role_address_from_path(path: Sequence[Any], *, root_role: str = "model") -> str:
    """Convert a JAX key path into a Feedbax role address."""

    return validate_role_address(".".join([root_role, *(_path_part(part) for part in path)]))


def template_array_roles(tree: Any, *, root_role: str = "model") -> list[str]:
    """Return deterministic role addresses for array leaves in ``tree``."""

    array_view = eqx.filter(tree, eqx.is_array)
    roles = [
        role_address_from_path(path, root_role=root_role)
        for path, leaf in jtu.tree_leaves_with_path(array_view)
        if leaf is not None
    ]
    if not roles:
        raise ArrayStoreValidationError("Template has no array leaves to materialize.")
    if len(set(roles)) != len(roles):
        raise ArrayStoreValidationError("Template array role addresses must be unique.")
    return sorted(roles)


def materialize_array_store(
    template: Any,
    store: ArrayStore,
    *,
    root_role: str = "model",
    exact_roles: bool = True,
) -> Any:
    """Fill ``template`` array leaves from a loaded role-addressed array store."""

    expected_roles = template_array_roles(template, root_role=root_role)
    if exact_roles:
        store.validate_roles(exact_roles=expected_roles)
    else:
        store.validate_roles(required_roles=expected_roles)

    used_roles: set[str] = set()
    array_view = eqx.filter(template, eqx.is_array)

    def replace(path, leaf):
        if not eqx.is_array(leaf):
            return leaf
        role = role_address_from_path(path, root_role=root_role)
        try:
            value = store.arrays[role]
        except KeyError as exc:
            raise ArrayStoreValidationError(
                f"Array store is missing role required by template: {role}"
            ) from exc
        _validate_template_leaf(role, leaf, value)
        used_roles.add(role)
        return jnp.asarray(value)

    materialized_arrays = jtu.tree_map_with_path(replace, array_view)
    materialized = eqx.combine(materialized_arrays, template)
    if exact_roles:
        unexpected_roles = sorted(set(store.arrays) - used_roles)
        if unexpected_roles:
            raise ArrayStoreValidationError(f"Array store has unexpected roles: {unexpected_roles}")
    return materialized


def materialize_model_artifact(
    manifest: ModelArtifactManifest | Path | str,
    template: Any,
    *,
    root: Path | str | None = None,
    root_role: str = "model",
    exact_roles: bool = True,
) -> Any:
    """Materialize a model artifact's parameter store into ``template``.

    This API intentionally does not instantiate a ``GraphSpec``. Callers that
    own domain-specific component builders provide the executable template; this
    function owns the Feedbax manifest/store validation and role materialization.

    Only current schema versions are materialized directly. Older artifacts
    must first pass through the Feedbax migration registry so load failures are
    deterministic rather than silent compatibility guesses.
    """

    manifest_obj, manifest_path = _load_model_manifest(manifest)
    _validate_supported_manifest(manifest_obj)
    if manifest_obj.parameter_store is None:
        raise ArrayStoreValidationError(
            f"Model artifact {manifest_obj.id!r} has no parameter_store."
        )
    _validate_supported_store_ref(manifest_obj)

    base = (
        Path(root) if root is not None else (manifest_path.parent if manifest_path else Path("."))
    )
    if manifest_obj.parameter_store.uri is None:
        raise ArrayStoreValidationError(
            f"Model artifact {manifest_obj.id!r} parameter_store has no URI."
        )

    store_path = Path(manifest_obj.parameter_store.uri)
    if not store_path.is_absolute():
        store_path = base / store_path

    digest = sha256_file(store_path)
    if manifest_obj.parameter_store.sha256 and digest != manifest_obj.parameter_store.sha256:
        raise ArrayStoreValidationError(
            f"Array store digest mismatch for {store_path}: "
            f"manifest={manifest_obj.parameter_store.sha256}, actual={digest}"
        )

    store = read_npz_array_store(store_path)
    _validate_supported_store_payload(store, manifest_obj)
    if store.payload.store_role != manifest_obj.parameter_store.role:
        raise ArrayStoreValidationError(
            f"Array store role mismatch: manifest={manifest_obj.parameter_store.role!r}, "
            f"store={store.payload.store_role!r}"
        )
    return materialize_array_store(
        template,
        store,
        root_role=root_role,
        exact_roles=exact_roles,
    )


def _load_model_manifest(
    manifest: ModelArtifactManifest | Path | str,
) -> tuple[ModelArtifactManifest, Path | None]:
    if isinstance(manifest, ModelArtifactManifest):
        return manifest, None
    path = Path(manifest)
    loaded = load_manifest(path)
    if not isinstance(loaded, ModelArtifactManifest):
        raise TypeError(f"Expected ModelArtifactManifest, got {type(loaded).__name__}.")
    return loaded, path


def _validate_supported_manifest(manifest: ModelArtifactManifest) -> None:
    if manifest.schema_version != SCHEMA_VERSION:
        raise UnsupportedArtifactSchemaError(
            f"Unsupported model artifact manifest schema {manifest.schema_version!r}; "
            f"expected {SCHEMA_VERSION!r}. Run a registered manifest migration before "
            "materializing this artifact."
        )


def _validate_supported_store_ref(manifest: ModelArtifactManifest) -> None:
    store_ref = manifest.parameter_store
    if store_ref is None:
        return
    if store_ref.schema_version != ARRAY_STORE_SCHEMA_VERSION:
        raise UnsupportedArtifactSchemaError(
            f"Unsupported parameter_store schema {store_ref.schema_version!r}; "
            f"expected {ARRAY_STORE_SCHEMA_VERSION!r}. Run a registered array-store "
            "migration before materializing this artifact."
        )
    if store_ref.storage_backend != NPZ_BACKEND_VERSION:
        raise UnsupportedArtifactSchemaError(
            f"Unsupported parameter_store backend {store_ref.storage_backend!r}; "
            f"expected {NPZ_BACKEND_VERSION!r}."
        )


def _validate_supported_store_payload(
    store: ArrayStore,
    manifest: ModelArtifactManifest,
) -> None:
    if store.payload.schema_version != ARRAY_STORE_SCHEMA_VERSION:
        raise UnsupportedArtifactSchemaError(
            f"Unsupported array store schema {store.payload.schema_version!r}; "
            f"expected {ARRAY_STORE_SCHEMA_VERSION!r}. Run a registered array-store "
            "migration before materializing artifact {manifest.id!r}."
        )
    if store.payload.storage_backend != NPZ_BACKEND_VERSION:
        raise UnsupportedArtifactSchemaError(
            f"Unsupported array store backend {store.payload.storage_backend!r}; "
            f"expected {NPZ_BACKEND_VERSION!r}."
        )


def _validate_template_leaf(role: str, template_leaf: Any, value: Any) -> None:
    if tuple(value.shape) != tuple(template_leaf.shape):
        raise ArrayStoreValidationError(
            f"Array role {role!r} shape mismatch for template leaf: "
            f"expected {tuple(template_leaf.shape)}, found {tuple(value.shape)}."
        )
    if str(value.dtype) != str(template_leaf.dtype):
        raise ArrayStoreValidationError(
            f"Array role {role!r} dtype mismatch for template leaf: "
            f"expected {template_leaf.dtype}, found {value.dtype}."
        )


def _path_part(part: Any) -> str:
    if hasattr(part, "name"):
        raw = str(part.name)
    elif hasattr(part, "key"):
        raw = str(part.key)
    elif hasattr(part, "idx"):
        raw = f"{int(part.idx):04d}"
    else:
        raw = str(part)
    return _sanitize_role_part(raw)


def _sanitize_role_part(value: str) -> str:
    sanitized = re.sub(r"[^A-Za-z0-9_.:/@+-]+", "_", value.strip())
    sanitized = sanitized.strip("._")
    return sanitized or "unnamed"
