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
from pydantic import Field

from feedbax.contracts.manifest import (
    EVALUATION_STATES_CONTAINER_SCHEMA_ID,
    EVALUATION_STATES_CONTAINER_SCHEMA_VERSION,
    ArtifactRef,
    StrictModel,
    safe_manifest_key,
    sha256_bytes,
    store_bytes_artifact,
)
from feedbax.contracts.migrations import UnsupportedSpecVersion, default_spec_registry


EVALUATION_STATES_ARTIFACT_ROLE = "evaluation_states"
EVALUATION_STATES_MEDIA_TYPE = "application/x-feedbax-states+npz"
EVALUATION_STATES_STORAGE_BACKEND = "npz.v1"
EVALUATION_STATES_METADATA_KEY = "__feedbax_evaluation_states__.json"
EVALUATION_STATES_ARRAY_KEY_TEMPLATE = "array_{index:06d}.npy"


class EvaluationStatesContainerError(ValueError):
    """Base class for evaluation-states container failures."""


class EvaluationStatesHashMismatch(EvaluationStatesContainerError):
    """Raised when artifact bytes do not match the pinned SHA-256 digest."""


class EvaluationStatesLeafError(EvaluationStatesContainerError):
    """Raised when a state pytree leaf is unsupported by the v1 container."""


class EvaluationStatesArrayRecord(StrictModel):
    """One array leaf encoded inside an evaluation-states container."""

    path: str
    storage_key: str
    dtype: str
    shape: tuple[int, ...]
    sha256: str


class EvaluationStatesContainerPayload(StrictModel):
    """Metadata envelope for the v1 evaluation-states NPZ container."""

    schema_id: Literal[
        "feedbax.manifest.evaluation_states_container"
    ] = EVALUATION_STATES_CONTAINER_SCHEMA_ID
    schema_version: Literal[
        "feedbax.manifest.evaluation_states_container.v1"
    ] = EVALUATION_STATES_CONTAINER_SCHEMA_VERSION
    storage_backend: Literal["npz.v1"] = EVALUATION_STATES_STORAGE_BACKEND
    treedef_proto_b64: str
    arrays: list[EvaluationStatesArrayRecord] = Field(default_factory=list)


def store_evaluation_states_artifact(
    states: Any,
    *,
    root: Path | str,
    manifest_id: str,
) -> ArtifactRef:
    """Store evaluation states as a governed artifact and return its ref."""
    data, payload = evaluation_states_container_bytes(states)
    return store_bytes_artifact(
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


def evaluation_states_container_bytes(
    states: Any,
) -> tuple[bytes, EvaluationStatesContainerPayload]:
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

    payload = EvaluationStatesContainerPayload(
        treedef_proto_b64=base64.b64encode(treedef.serialize_using_proto()).decode("ascii"),
        arrays=records,
    )
    return _npz_bytes(payload, arrays), payload


def load_evaluation_states_container_bytes(data: bytes) -> Any:
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
        payload = EvaluationStatesContainerPayload.model_validate(payload_data)

        arrays: list[np.ndarray] = []
        seen = {EVALUATION_STATES_METADATA_KEY}
        for record in payload.arrays:
            if record.storage_key not in names:
                raise EvaluationStatesContainerError(
                    f"Evaluation states leaf {record.path} is missing storage key "
                    f"{record.storage_key!r}."
                )
            with archive.open(record.storage_key, mode="r") as member:
                array = np.load(member, allow_pickle=False)
            seen.add(record.storage_key)
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
            arrays.append(array)

        extra = sorted(names - seen)
        if extra:
            raise EvaluationStatesContainerError(
                f"Evaluation states container has unknown members: {extra}"
            )

    treedef_proto = base64.b64decode(payload.treedef_proto_b64.encode("ascii"))
    treedef = jtu.PyTreeDef.deserialize_using_proto(jtu.default_registry, treedef_proto)
    return treedef.unflatten(arrays)


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
    default_spec_registry.migrate(
        "EvaluationStatesContainer",
        payload,
        source_version=schema_version if isinstance(schema_version, str) else None,
    )


def _array_digest(array: np.ndarray) -> str:
    contiguous = np.ascontiguousarray(array)
    return sha256_bytes(contiguous.tobytes(order="C"))


def _npz_bytes(
    payload: EvaluationStatesContainerPayload,
    arrays: dict[str, np.ndarray],
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
