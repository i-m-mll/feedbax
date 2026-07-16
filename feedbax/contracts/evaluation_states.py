"""Versioned NPZ custody for evaluation-state pytrees."""

from __future__ import annotations

import base64
import io
import json
import zipfile
from pathlib import Path
from typing import Any, Literal

import jax
import jax.tree as jt
import jax.tree_util as jtu
import numpy as np
from pydantic import Field, ValidationError

from feedbax.contracts.manifest import (
    EVALUATION_STATES_CONTAINER_SCHEMA_ID,
    EVALUATION_STATES_CONTAINER_SCHEMA_VERSION,
    EVALUATION_STATES_CONTAINER_SCHEMA_VERSION_V1,
    EVALUATION_STATES_CONTAINER_SCHEMA_VERSION_V3,
    EVALUATION_STATES_STORAGE_BACKEND_V3,
    ArtifactRef,
    StrictModel,
    safe_manifest_key,
    sha256_bytes,
    store_bytes_artifact,
)
from feedbax.contracts.migrations import UnsupportedSpecVersion


EVALUATION_STATES_ARTIFACT_ROLE = "evaluation_states"
EVALUATION_STATES_MEDIA_TYPE = "application/x-feedbax-states+npz"
EVALUATION_STATES_STORAGE_BACKEND_V1 = "npz.v1"
EVALUATION_STATES_STORAGE_BACKEND = "npz.v2"
EVALUATION_STATES_METADATA_KEY = "__feedbax_evaluation_states__.json"
EVALUATION_STATES_METADATA_VALUES_KEY = "__feedbax_evaluation_states_metadata__.json"
EVALUATION_STATES_ARRAY_KEY_TEMPLATE = "array_{index:06d}.npy"


class EvaluationStatesContainerError(ValueError):
    """Base class for evaluation-states container failures."""


class EvaluationStatesHashMismatch(EvaluationStatesContainerError):
    """Raised when artifact bytes do not match the pinned SHA-256 digest."""


class EvaluationStatesCustodyUnavailable(EvaluationStatesContainerError):
    """Raised when durable evaluation-state custody cannot be authenticated."""


class EvaluationStatesSchemaMismatch(EvaluationStatesContainerError):
    """Raised when durable evaluation-state schema evidence is unsupported."""


class EvaluationStatesProvenanceMismatch(EvaluationStatesContainerError):
    """Raised when durable evaluation-state bytes do not name their supplier."""


class EvaluationStatesSizeMismatch(
    EvaluationStatesHashMismatch,
    EvaluationStatesCustodyUnavailable,
):
    """Raised when authenticated bytes disagree with their declared size."""


class EvaluationStatesLeafError(EvaluationStatesContainerError):
    """Raised when a state pytree leaf is unsupported by the v1 container."""


class EvaluationStatesArrayRecord(StrictModel):
    """One array leaf encoded inside an evaluation-states container."""

    path: str
    storage_key: str
    dtype: str
    shape: tuple[int, ...]
    sha256: str


class EvaluationStatesContainerPayloadV1(StrictModel):
    """Metadata envelope for the v1 evaluation-states NPZ container."""

    schema_id: Literal[
        "feedbax.manifest.evaluation_states_container"
    ] = EVALUATION_STATES_CONTAINER_SCHEMA_ID
    schema_version: Literal[
        "feedbax.manifest.evaluation_states_container.v1"
    ] = EVALUATION_STATES_CONTAINER_SCHEMA_VERSION_V1
    storage_backend: Literal["npz.v1"] = EVALUATION_STATES_STORAGE_BACKEND_V1
    treedef_proto_b64: str
    arrays: list[EvaluationStatesArrayRecord] = Field(default_factory=list)


class EvaluationStatesLeafRecord(StrictModel):
    """One v2 pytree leaf encoded as an array member or metadata JSON value."""

    path: str
    kind: Literal["array", "metadata"]
    sha256: str
    storage_key: str | None = None
    dtype: str | None = None
    shape: tuple[int, ...] | None = None


class EvaluationStatesContainerPayloadV2(StrictModel):
    """Metadata envelope for the v2 mixed evaluation-states NPZ container."""

    schema_id: Literal[
        "feedbax.manifest.evaluation_states_container"
    ] = EVALUATION_STATES_CONTAINER_SCHEMA_ID
    schema_version: Literal[
        "feedbax.manifest.evaluation_states_container.v2"
    ] = EVALUATION_STATES_CONTAINER_SCHEMA_VERSION
    storage_backend: Literal["npz.v2"] = EVALUATION_STATES_STORAGE_BACKEND
    treedef_proto_b64: str
    leaves: list[EvaluationStatesLeafRecord] = Field(default_factory=list)
    metadata_sha256: str | None = None


class EvaluationStatesContainerPayloadV3(StrictModel):
    """Metadata envelope for the v3 leaves-only evaluation-states container."""

    schema_id: Literal[
        "feedbax.manifest.evaluation_states_container"
    ] = EVALUATION_STATES_CONTAINER_SCHEMA_ID
    schema_version: Literal[
        "feedbax.manifest.evaluation_states_container.v3"
    ] = EVALUATION_STATES_CONTAINER_SCHEMA_VERSION_V3
    storage_backend: Literal["npz.v3"] = EVALUATION_STATES_STORAGE_BACKEND_V3
    structure_fingerprint: str = Field(
        description=(
            "SHA-256 of canonical JSON for a preorder PyTreeDef walk; each node records "
            "its module-qualified node type (or '<leaf>') and a deterministic repr of "
            "auxiliary data with mapping/set entries sorted recursively."
        )
    )
    leaves: list[EvaluationStatesLeafRecord] = Field(default_factory=list)
    metadata_sha256: str | None = None


EvaluationStatesContainerPayload = EvaluationStatesContainerPayloadV2


def store_evaluation_states_artifact(
    states: Any,
    *,
    root: Path | str,
    manifest_id: str,
) -> ArtifactRef:
    """Store evaluation states as a governed artifact and return its ref."""
    treedef = jt.structure(states)
    writer = (
        evaluation_states_container_bytes
        if _treedef_is_proto_serializable(treedef)
        else evaluation_states_container_bytes_v3
    )
    data, payload = writer(states)
    ref = store_bytes_artifact(
        data,
        root=root,
        role=EVALUATION_STATES_ARTIFACT_ROLE,
        logical_name=f"{safe_manifest_key(manifest_id)}.states.npz",
        media_type=EVALUATION_STATES_MEDIA_TYPE,
        suffix=".npz",
        metadata={
            "schema_id": payload.schema_id,
            "schema_version": payload.schema_version,
            "storage_backend": payload.storage_backend,
            "manifest_id": manifest_id,
        },
    )
    return ref.model_copy(update={"uri": None})


def load_evaluation_states_artifact(
    artifact: ArtifactRef,
    *,
    root: Path | str,
) -> Any:
    """Load and verify evaluation states from an ``evaluation_states`` artifact."""
    path = _artifact_path(artifact, root=Path(root))
    data = path.read_bytes()
    if artifact.sha256 is None:
        raise EvaluationStatesHashMismatch(
            f"Evaluation states artifact {artifact.logical_name!r} has no sha256."
        )
    digest = sha256_bytes(data)
    if digest != artifact.sha256:
        raise EvaluationStatesHashMismatch(
            "Evaluation states artifact SHA-256 mismatch: "
            f"logical_name={artifact.logical_name!r}, expected={artifact.sha256!r}, "
            f"computed={digest!r}"
        )
    return load_evaluation_states_container_bytes(data)


def load_authenticated_evaluation_states_artifact(
    artifact: ArtifactRef,
    *,
    root: Path | str | None = None,
    manifest_id: str,
    data: bytes | None = None,
    structure: jtu.PyTreeDef | None = None,
) -> Any:
    """Load states only after authenticating custody, schema, and provenance evidence."""

    if artifact.role != EVALUATION_STATES_ARTIFACT_ROLE:
        raise EvaluationStatesProvenanceMismatch(
            f"Expected artifact role {EVALUATION_STATES_ARTIFACT_ROLE!r}; got {artifact.role!r}."
        )
    if not artifact.artifact_id or not artifact.sha256:
        raise EvaluationStatesCustodyUnavailable(
            "Durable evaluation states require artifact_id and sha256 custody identity."
        )
    if artifact.size_bytes is None or artifact.size_bytes < 0:
        raise EvaluationStatesCustodyUnavailable(
            "Durable evaluation states require a nonnegative declared byte size."
        )
    expected_artifact_id = f"artifact://sha256/{artifact.sha256}"
    if artifact.artifact_id != expected_artifact_id:
        raise EvaluationStatesCustodyUnavailable(
            "Evaluation-state artifact identity disagrees with its SHA-256: "
            f"artifact_id={artifact.artifact_id!r}, expected={expected_artifact_id!r}."
        )
    if artifact.media_type != EVALUATION_STATES_MEDIA_TYPE:
        raise EvaluationStatesSchemaMismatch(
            "Evaluation-state artifact has unsupported media type: "
            f"{artifact.media_type!r}; expected {EVALUATION_STATES_MEDIA_TYPE!r}."
        )

    metadata = artifact.metadata
    if metadata.get("manifest_id") != manifest_id:
        raise EvaluationStatesProvenanceMismatch(
            "Evaluation-state artifact provenance does not name its supplying manifest: "
            f"metadata.manifest_id={metadata.get('manifest_id')!r}, expected={manifest_id!r}."
        )
    schema_version = metadata.get("schema_version")
    expected_storage = {
        EVALUATION_STATES_CONTAINER_SCHEMA_VERSION_V1: EVALUATION_STATES_STORAGE_BACKEND_V1,
        EVALUATION_STATES_CONTAINER_SCHEMA_VERSION: EVALUATION_STATES_STORAGE_BACKEND,
        EVALUATION_STATES_CONTAINER_SCHEMA_VERSION_V3: EVALUATION_STATES_STORAGE_BACKEND_V3,
    }
    if (
        metadata.get("schema_id") != EVALUATION_STATES_CONTAINER_SCHEMA_ID
        or schema_version not in expected_storage
        or metadata.get("storage_backend") != expected_storage.get(schema_version)
    ):
        raise EvaluationStatesSchemaMismatch(
            "Evaluation-state artifact schema/storage metadata is unsupported: "
            f"schema_id={metadata.get('schema_id')!r}, schema_version={schema_version!r}, "
            f"storage_backend={metadata.get('storage_backend')!r}."
        )

    if data is None:
        if root is None:
            raise EvaluationStatesCustodyUnavailable(
                "Evaluation-state artifact requires explicit custody root or pre-read bytes."
            )
        try:
            path = _artifact_path(artifact, root=Path(root))
            data = path.read_bytes()
        except (FileNotFoundError, OSError) as exc:
            raise EvaluationStatesCustodyUnavailable(
                f"Evaluation-state artifact bytes are unavailable: {exc}"
            ) from exc
    if len(data) != artifact.size_bytes:
        raise EvaluationStatesSizeMismatch(
            "Evaluation states artifact byte-size mismatch: "
            f"logical_name={artifact.logical_name!r}, expected={artifact.size_bytes}, "
            f"computed={len(data)}"
        )
    digest = sha256_bytes(data)
    if digest != artifact.sha256:
        raise EvaluationStatesHashMismatch(
            "Evaluation states artifact SHA-256 mismatch: "
            f"logical_name={artifact.logical_name!r}, expected={artifact.sha256!r}, "
            f"computed={digest!r}"
        )

    try:
        return load_evaluation_states_container_bytes(data, structure=structure)
    except UnsupportedSpecVersion as exc:
        raise EvaluationStatesSchemaMismatch(str(exc)) from exc
    except ValidationError as exc:
        schema_fields = {"schema_id", "schema_version", "storage_backend"}
        error_fields = {error["loc"][0] for error in exc.errors() if error["loc"]}
        error_type = (
            EvaluationStatesSchemaMismatch
            if error_fields.intersection(schema_fields)
            else EvaluationStatesCustodyUnavailable
        )
        raise error_type(f"Evaluation-state container header is invalid: {exc}") from exc
    except (
        zipfile.BadZipFile,
        json.JSONDecodeError,
        UnicodeDecodeError,
        KeyError,
        EvaluationStatesContainerError,
    ) as exc:
        raise EvaluationStatesCustodyUnavailable(
            f"Evaluation-state container bytes could not be verified: {exc}"
        ) from exc


def evaluation_states_container_bytes(
    states: Any,
) -> tuple[bytes, EvaluationStatesContainerPayload]:
    """Serialize a mixed array/JSON-metadata pytree into deterministic v2 bytes."""
    path_leaves, treedef = jt.flatten_with_path(states)
    if not path_leaves:
        raise EvaluationStatesLeafError(
            "states_custody='durable' requires a non-empty pytree of array or "
            "JSON-serializable metadata leaves."
        )

    arrays, leaves, metadata_bytes = _encode_mixed_leaves(path_leaves)
    payload = EvaluationStatesContainerPayloadV2(
        treedef_proto_b64=base64.b64encode(treedef.serialize_using_proto()).decode("ascii"),
        leaves=leaves,
        metadata_sha256=sha256_bytes(metadata_bytes) if metadata_bytes is not None else None,
    )
    return _npz_bytes(payload, arrays, metadata_bytes=metadata_bytes), payload


def evaluation_states_container_bytes_v3(
    states: Any,
) -> tuple[bytes, EvaluationStatesContainerPayloadV3]:
    """Serialize mixed leaves without embedding a caller-specific pytree definition."""
    path_leaves, treedef = jt.flatten_with_path(states)
    if not path_leaves:
        raise EvaluationStatesLeafError(
            "states_custody='durable' requires a non-empty pytree of array or "
            "JSON-serializable metadata leaves."
        )
    arrays, leaves, metadata_bytes = _encode_mixed_leaves(path_leaves)
    payload = EvaluationStatesContainerPayloadV3(
        structure_fingerprint=_treedef_structure_fingerprint(treedef),
        leaves=leaves,
        metadata_sha256=sha256_bytes(metadata_bytes) if metadata_bytes is not None else None,
    )
    return _npz_bytes(payload, arrays, metadata_bytes=metadata_bytes), payload


def _treedef_is_proto_serializable(treedef: jtu.PyTreeDef) -> bool:
    """Return whether JAX's proto codec supports this pytree structure."""
    # This match is coupled to JAX's current error wording; unknown wording re-raises so
    # writer selection fails safe and loudly when that upstream contract changes.
    try:
        treedef.serialize_using_proto()
    except ValueError as exc:
        if "User-defined nodes are not supported" in str(exc):
            return False
        raise
    return True


def evaluation_states_container_bytes_v1(
    states: Any,
) -> tuple[bytes, EvaluationStatesContainerPayloadV1]:
    """Serialize an array-only pytree into deterministic NPZ container bytes."""
    path_leaves, treedef = jt.flatten_with_path(states)
    if not path_leaves:
        raise EvaluationStatesLeafError(
            "states_custody='durable' requires a non-empty pytree of array leaves."
        )

    arrays: dict[str, np.ndarray] = {}
    records: list[EvaluationStatesArrayRecord] = []
    for index, (path, leaf) in enumerate(path_leaves):
        leaf_path = _leaf_path(path)
        if not isinstance(leaf, (jax.Array, np.ndarray)):
            raise EvaluationStatesLeafError(
                "Evaluation states container v1 only supports array leaves; "
                f"unsupported leaf at {leaf_path}: {type(leaf).__name__}"
            )
        array = np.asarray(leaf)
        storage_key = EVALUATION_STATES_ARRAY_KEY_TEMPLATE.format(index=index)
        arrays[storage_key] = array
        records.append(
            EvaluationStatesArrayRecord(
                path=leaf_path,
                storage_key=storage_key,
                dtype=str(array.dtype),
                shape=tuple(int(dim) for dim in array.shape),
                sha256=_array_digest(array),
            )
        )

    payload = EvaluationStatesContainerPayloadV1(
        treedef_proto_b64=base64.b64encode(treedef.serialize_using_proto()).decode("ascii"),
        arrays=records,
    )
    return _npz_bytes(payload, arrays), payload


def load_evaluation_states_container_bytes(
    data: bytes,
    *,
    structure: jtu.PyTreeDef | None = None,
) -> Any:
    """Load evaluation states from verified NPZ container bytes."""
    with zipfile.ZipFile(io.BytesIO(data), mode="r") as archive:
        names = set(archive.namelist())
        if EVALUATION_STATES_METADATA_KEY not in names:
            raise EvaluationStatesContainerError(
                "Evaluation states container is missing metadata member "
                f"{EVALUATION_STATES_METADATA_KEY!r}."
            )
        raw_payload = archive.read(EVALUATION_STATES_METADATA_KEY)
        payload_data = json.loads(raw_payload.decode("utf-8"))
        _validate_payload_version(payload_data)
        schema_version = payload_data.get("schema_version")
        if schema_version == EVALUATION_STATES_CONTAINER_SCHEMA_VERSION_V1:
            payload = EvaluationStatesContainerPayloadV1.model_validate(payload_data)
            leaves = _load_v1_leaves(archive, names, payload)
            seen = {EVALUATION_STATES_METADATA_KEY} | {
                record.storage_key for record in payload.arrays
            }
        elif schema_version == EVALUATION_STATES_CONTAINER_SCHEMA_VERSION:
            payload = EvaluationStatesContainerPayloadV2.model_validate(payload_data)
            leaves, seen = _load_v2_leaves(archive, names, payload)
        elif schema_version == EVALUATION_STATES_CONTAINER_SCHEMA_VERSION_V3:
            payload = EvaluationStatesContainerPayloadV3.model_validate(payload_data)
            if structure is None:
                raise EvaluationStatesContainerError(
                    "Evaluation states container v3 requires caller-supplied structure."
                )
            if structure.num_leaves != len(payload.leaves):
                raise EvaluationStatesContainerError(
                    "Evaluation states container v3 leaf count does not match caller structure: "
                    f"stored={len(payload.leaves)}, structure={structure.num_leaves}."
                )
            structure_paths = _treedef_leaf_paths(structure)
            stored_paths = [record.path for record in payload.leaves]
            if stored_paths != structure_paths:
                raise EvaluationStatesContainerError(
                    "Evaluation states container v3 leaf paths do not match caller structure: "
                    f"stored={stored_paths!r}, structure={structure_paths!r}."
                )
            structure_fingerprint = _treedef_structure_fingerprint(structure)
            if structure_fingerprint != payload.structure_fingerprint:
                raise EvaluationStatesContainerError(
                    "Evaluation states container v3 structure fingerprint does not match "
                    "caller structure."
                )
            leaves, seen = _load_v2_leaves(archive, names, payload, version_label="v3")
        else:  # pragma: no cover - _validate_payload_version raises first.
            raise UnsupportedSpecVersion(
                f"Unsupported evaluation states container version {schema_version!r}."
            )

        extra = sorted(names - seen)
        if extra:
            raise EvaluationStatesContainerError(
                f"Evaluation states container has unknown members: {extra}"
            )

    if schema_version == EVALUATION_STATES_CONTAINER_SCHEMA_VERSION_V3:
        assert structure is not None
        return structure.unflatten(leaves)
    treedef_proto = base64.b64decode(payload.treedef_proto_b64.encode("ascii"))
    treedef = jtu.PyTreeDef.deserialize_using_proto(jtu.default_registry, treedef_proto)
    if structure is not None and treedef != structure:
        raise EvaluationStatesContainerError(
            "Evaluation states container pytree does not match caller-supplied structure."
        )
    return treedef.unflatten(leaves)


def _load_v1_leaves(
    archive: zipfile.ZipFile,
    names: set[str],
    payload: EvaluationStatesContainerPayloadV1,
) -> list[np.ndarray]:
    arrays: list[np.ndarray] = []
    for record in payload.arrays:
        arrays.append(_load_array_leaf(archive, names, record))
    return arrays


def _load_v2_leaves(
    archive: zipfile.ZipFile,
    names: set[str],
    payload: EvaluationStatesContainerPayloadV2 | EvaluationStatesContainerPayloadV3,
    *,
    version_label: str = "v2",
) -> tuple[list[Any], set[str]]:
    seen = {EVALUATION_STATES_METADATA_KEY}
    metadata_by_index: dict[int, Any] = {}
    has_metadata = any(record.kind == "metadata" for record in payload.leaves)
    if has_metadata:
        if EVALUATION_STATES_METADATA_VALUES_KEY not in names:
            raise EvaluationStatesContainerError(
                f"Evaluation states container {version_label} is missing metadata values member "
                f"{EVALUATION_STATES_METADATA_VALUES_KEY!r}."
            )
        metadata_bytes = archive.read(EVALUATION_STATES_METADATA_VALUES_KEY)
        seen.add(EVALUATION_STATES_METADATA_VALUES_KEY)
        if payload.metadata_sha256 is None:
            raise EvaluationStatesContainerError(
                f"Evaluation states container {version_label} metadata section has no SHA-256 digest."
            )
        digest = sha256_bytes(metadata_bytes)
        if digest != payload.metadata_sha256:
            raise EvaluationStatesContainerError(
                f"Evaluation states container {version_label} metadata section digest mismatch."
            )
        metadata_records = json.loads(metadata_bytes.decode("utf-8"))
        if not isinstance(metadata_records, list):
            raise EvaluationStatesContainerError(
                f"Evaluation states container {version_label} metadata section must be a list."
            )
        for item in metadata_records:
            if not isinstance(item, dict):
                raise EvaluationStatesContainerError(
                    f"Evaluation states container {version_label} metadata entries must be objects."
                )
            index = item.get("index")
            path = item.get("path")
            if not isinstance(index, int) or not isinstance(path, str) or "value" not in item:
                raise EvaluationStatesContainerError(
                    f"Evaluation states container {version_label} metadata entry is malformed."
                )
            if index in metadata_by_index:
                raise EvaluationStatesContainerError(
                    f"Evaluation states container {version_label} metadata duplicates leaf {index}."
                )
            if index < 0 or index >= len(payload.leaves):
                raise EvaluationStatesContainerError(
                    f"Evaluation states container {version_label} metadata index {index} is out of range."
                )
            if payload.leaves[index].path != path:
                raise EvaluationStatesContainerError(
                    f"Evaluation states container {version_label} metadata path mismatch: "
                    f"index={index}, expected={payload.leaves[index].path!r}, found={path!r}."
                )
            metadata_by_index[index] = item["value"]
    elif EVALUATION_STATES_METADATA_VALUES_KEY in names:
        raise EvaluationStatesContainerError(
            f"Evaluation states container {version_label} has a metadata values member but no "
            "metadata leaves."
        )
    elif payload.metadata_sha256 is not None:
        raise EvaluationStatesContainerError(
            f"Evaluation states container {version_label} has a metadata digest but no metadata leaves."
        )

    leaves: list[Any] = []
    for index, record in enumerate(payload.leaves):
        if record.kind == "array":
            if record.storage_key is None or record.dtype is None or record.shape is None:
                raise EvaluationStatesContainerError(
                    f"Evaluation states array leaf {record.path} is missing metadata."
                )
            array_record = EvaluationStatesArrayRecord(
                path=record.path,
                storage_key=record.storage_key,
                dtype=record.dtype,
                shape=record.shape,
                sha256=record.sha256,
            )
            leaves.append(_load_array_leaf(archive, names, array_record))
            seen.add(record.storage_key)
            continue

        if record.storage_key is not None or record.dtype is not None or record.shape is not None:
            raise EvaluationStatesContainerError(
                f"Evaluation states metadata leaf {record.path} has array metadata."
            )
        if index not in metadata_by_index:
            raise EvaluationStatesContainerError(
                f"Evaluation states metadata leaf {record.path} is missing a value."
            )
        value = metadata_by_index[index]
        digest = sha256_bytes(_canonical_json_bytes(value))
        if digest != record.sha256:
            raise EvaluationStatesContainerError(
                f"Evaluation states metadata leaf {record.path} digest mismatch."
            )
        leaves.append(value)

    if len(metadata_by_index) != sum(record.kind == "metadata" for record in payload.leaves):
        raise EvaluationStatesContainerError(
            f"Evaluation states container {version_label} metadata section has unreferenced entries."
        )
    return leaves, seen


def _load_array_leaf(
    archive: zipfile.ZipFile,
    names: set[str],
    record: EvaluationStatesArrayRecord,
) -> np.ndarray:
    if record.storage_key not in names:
        raise EvaluationStatesContainerError(
            f"Evaluation states leaf {record.path} is missing storage key "
            f"{record.storage_key!r}."
        )
    with archive.open(record.storage_key, mode="r") as member:
        array = np.load(member, allow_pickle=False)
    if str(array.dtype) != record.dtype or tuple(array.shape) != record.shape:
        raise EvaluationStatesContainerError(
            f"Evaluation states leaf {record.path} metadata mismatch: "
            f"expected dtype={record.dtype}, shape={record.shape}; "
            f"found dtype={array.dtype}, shape={tuple(array.shape)}."
        )
    digest = _array_digest(array)
    if digest != record.sha256:
        raise EvaluationStatesContainerError(
            f"Evaluation states leaf {record.path} digest mismatch."
        )
    return array


def _artifact_path(artifact: ArtifactRef, *, root: Path) -> Path:
    if artifact.uri:
        uri_path = Path(artifact.uri)
        return uri_path if uri_path.is_absolute() else root / uri_path
    relative_path = artifact.metadata.get("relative_path")
    if isinstance(relative_path, str) and relative_path:
        return root / relative_path
    raise FileNotFoundError(
        f"Evaluation states artifact {artifact.logical_name!r} has no uri or relative_path."
    )


def _validate_payload_version(payload: dict[str, Any]) -> None:
    schema_id = payload.get("schema_id")
    if schema_id != EVALUATION_STATES_CONTAINER_SCHEMA_ID:
        raise UnsupportedSpecVersion(
            "Unsupported evaluation states container schema identity: "
            f"schema_id={schema_id!r}, expected={EVALUATION_STATES_CONTAINER_SCHEMA_ID!r}"
        )
    schema_version = payload.get("schema_version")
    accepted_versions = {
        EVALUATION_STATES_CONTAINER_SCHEMA_VERSION_V1,
        EVALUATION_STATES_CONTAINER_SCHEMA_VERSION,
        EVALUATION_STATES_CONTAINER_SCHEMA_VERSION_V3,
    }
    if schema_version not in accepted_versions:
        raise UnsupportedSpecVersion(
            "Unsupported EvaluationStatesContainer version: "
            f"schema_version={schema_version!r}, accepted={sorted(accepted_versions)!r}."
        )


def _encode_mixed_leaves(
    path_leaves: list[tuple[tuple[Any, ...], Any]],
) -> tuple[dict[str, np.ndarray], list[EvaluationStatesLeafRecord], bytes | None]:
    arrays: dict[str, np.ndarray] = {}
    leaves: list[EvaluationStatesLeafRecord] = []
    metadata_values: list[dict[str, Any]] = []
    for index, (path, leaf) in enumerate(path_leaves):
        leaf_path = _leaf_path(path)
        if isinstance(leaf, (jax.Array, np.ndarray)):
            array = np.asarray(leaf)
            storage_key = EVALUATION_STATES_ARRAY_KEY_TEMPLATE.format(index=index)
            arrays[storage_key] = array
            leaves.append(
                EvaluationStatesLeafRecord(
                    path=leaf_path,
                    kind="array",
                    storage_key=storage_key,
                    dtype=str(array.dtype),
                    shape=tuple(int(dim) for dim in array.shape),
                    sha256=_array_digest(array),
                )
            )
            continue
        value_bytes = _canonical_metadata_leaf_bytes(leaf, leaf_path)
        metadata_values.append(
            {"index": index, "path": leaf_path, "value": json.loads(value_bytes)}
        )
        leaves.append(
            EvaluationStatesLeafRecord(
                path=leaf_path,
                kind="metadata",
                sha256=sha256_bytes(value_bytes),
            )
        )
    metadata_bytes = (
        _canonical_json_bytes(metadata_values) + b"\n" if metadata_values else None
    )
    return arrays, leaves, metadata_bytes


def _treedef_leaf_paths(structure: jtu.PyTreeDef) -> list[str]:
    prototype = structure.unflatten([object() for _ in range(structure.num_leaves)])
    path_leaves, derived = jt.flatten_with_path(prototype)
    if derived != structure:
        raise EvaluationStatesContainerError(
            "Caller-supplied evaluation states structure cannot be flattened consistently."
        )
    return [_leaf_path(path) for path, _ in path_leaves]


def _treedef_structure_fingerprint(structure: jtu.PyTreeDef) -> str:
    """Hash each node's type, stable aux representation, and arity in preorder."""
    nodes: list[dict[str, str | int]] = []

    def walk(node: jtu.PyTreeDef) -> None:
        node_data = node.node_data()
        if node_data is None:
            node_type, aux_data = "<leaf>", "None"
        else:
            node_class, aux = node_data
            node_type = f"{node_class.__module__}.{node_class.__qualname__}"
            aux_data = _stable_aux_repr(aux)
        child_count = len(node.children())
        nodes.append({"node_type": node_type, "aux_data_repr": aux_data, "child_count": child_count})
        for child in node.children():
            walk(child)

    walk(structure)
    return sha256_bytes(_canonical_json_bytes(nodes))


def _stable_aux_repr(value: Any) -> str:
    """Return stable aux text; address-bearing custom reprs fail safe by rejecting loads."""
    if isinstance(value, dict):
        items = sorted((_stable_aux_repr(key), _stable_aux_repr(item)) for key, item in value.items())
        return f"dict({items!r})"
    if isinstance(value, (set, frozenset)):
        return f"{type(value).__name__}({sorted(_stable_aux_repr(item) for item in value)!r})"
    if isinstance(value, (list, tuple)):
        items = [_stable_aux_repr(item) for item in value]
        return f"{type(value).__name__}({items!r})"
    if isinstance(value, type):
        return f"type({value.__module__}.{value.__qualname__})"
    return repr(value)


def _array_digest(array: np.ndarray) -> str:
    contiguous = np.ascontiguousarray(array)
    return sha256_bytes(contiguous.tobytes(order="C"))


def _canonical_metadata_leaf_bytes(value: Any, leaf_path: str) -> bytes:
    try:
        return _canonical_json_bytes(value)
    except (TypeError, ValueError) as exc:
        raise EvaluationStatesLeafError(
            "Evaluation states container v2 only supports array leaves and "
            "JSON-serializable metadata leaves; unsupported leaf at "
            f"{leaf_path}: {type(value).__name__}"
        ) from exc


def _canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")


def _npz_bytes(
    payload: (
        EvaluationStatesContainerPayloadV1
        | EvaluationStatesContainerPayloadV2
        | EvaluationStatesContainerPayloadV3
    ),
    arrays: dict[str, np.ndarray],
    *,
    metadata_bytes: bytes | None = None,
) -> bytes:
    output = io.BytesIO()
    with zipfile.ZipFile(output, mode="w", compression=zipfile.ZIP_DEFLATED) as archive:
        _write_zip_member(
            archive,
            EVALUATION_STATES_METADATA_KEY,
            json.dumps(
                payload.model_dump(mode="json", exclude_none=True),
                indent=2,
                sort_keys=True,
            ).encode("utf-8") + b"\n",
        )
        if metadata_bytes is not None:
            _write_zip_member(
                archive,
                EVALUATION_STATES_METADATA_VALUES_KEY,
                metadata_bytes,
            )
        for key in sorted(arrays):
            array_buffer = io.BytesIO()
            np.lib.format.write_array(array_buffer, arrays[key], allow_pickle=False)
            _write_zip_member(archive, key, array_buffer.getvalue())
    return output.getvalue()


def _write_zip_member(archive: zipfile.ZipFile, name: str, data: bytes) -> None:
    info = zipfile.ZipInfo(filename=name, date_time=(1980, 1, 1, 0, 0, 0))
    info.compress_type = zipfile.ZIP_DEFLATED
    archive.writestr(info, data)


def _leaf_path(path: tuple[Any, ...]) -> str:
    if not path:
        return "<root>"
    parts: list[str] = []
    for entry in path:
        key = getattr(entry, "key", None)
        if key is not None:
            parts.append(f"[{key!r}]")
            continue
        index = getattr(entry, "idx", None)
        if index is not None:
            parts.append(f"[{index}]")
            continue
        name = getattr(entry, "name", None)
        if name is not None:
            parts.append(f".{name}")
            continue
        parts.append(f"[{entry!r}]")
    return "".join(parts)
