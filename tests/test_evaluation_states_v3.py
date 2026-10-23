from __future__ import annotations

import hashlib
import io
import json
import os
import subprocess
import sys
import zipfile
from pathlib import Path
from typing import NamedTuple

import jax.tree as jt
import numpy as np
import pytest

from feedbax.contracts.evaluation_states import (
    EVALUATION_STATES_ARTIFACT_ROLE,
    EVALUATION_STATES_MEDIA_TYPE,
    EVALUATION_STATES_METADATA_KEY,
    EVALUATION_STATES_STORAGE_BACKEND_V3,
    EvaluationStatesContainerError, evaluation_states_container_bytes,
    evaluation_states_container_bytes_v3, load_authenticated_evaluation_states_artifact,
    load_evaluation_states_container_bytes, store_evaluation_states_artifact,
)
from feedbax.contracts.artifact_store import store_bytes_artifact
from feedbax.contracts.manifest import (
    EVALUATION_STATES_CONTAINER_SCHEMA_VERSION,
    EVALUATION_STATES_CONTAINER_SCHEMA_VERSION_V3,
)
from feedbax.contracts.migrations import UnsupportedSpecVersion


class Inner(NamedTuple):
    hidden: object


class Memory(NamedTuple):
    step: object
    inner: Inner


class Trajectory(NamedTuple):
    states: object
    commands: object
    feedback: object
    final_memory: Memory


def _states() -> Trajectory:
    return Trajectory(
        states=np.arange(6, dtype=np.float32).reshape(2, 3),
        commands=np.arange(4, dtype=np.float32),
        feedback=np.asarray([1, 0], dtype=np.int32),
        final_memory=Memory(
            step=np.asarray(12, dtype=np.int32),
            inner=Inner(hidden=np.asarray([0.25, 0.5], dtype=np.float32)),
        ),
    )


def _rewrite_container(data: bytes, mutate, *, extra: bool = False) -> bytes:
    output = io.BytesIO()
    with (
        zipfile.ZipFile(io.BytesIO(data), mode="r") as source,
        zipfile.ZipFile(output, mode="w", compression=zipfile.ZIP_DEFLATED) as target,
    ):
        for name in source.namelist():
            member = source.read(name)
            if name == EVALUATION_STATES_METADATA_KEY:
                payload = json.loads(member)
                mutate(payload)
                member = json.dumps(payload, indent=2, sort_keys=True).encode() + b"\n"
            info = zipfile.ZipInfo(name, date_time=(1980, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            target.writestr(info, member)
        if extra:
            target.writestr("unexpected.bin", b"nope")
    return output.getvalue()


def test_v3_round_trip_is_deterministic_and_preserves_namedtuple_types() -> None:
    states = _states()
    structure = jt.structure(states)
    with pytest.raises(ValueError, match="User-defined nodes are not supported"):
        structure.serialize_using_proto()
    data, payload = evaluation_states_container_bytes_v3(states)
    repeated, repeated_payload = evaluation_states_container_bytes_v3(states)
    loaded = load_evaluation_states_container_bytes(data, structure=structure)

    assert data == repeated
    assert payload == repeated_payload
    assert payload.schema_version == EVALUATION_STATES_CONTAINER_SCHEMA_VERSION_V3
    assert payload.structure_fingerprint == repeated_payload.structure_fingerprint
    assert isinstance(loaded, Trajectory)
    assert isinstance(loaded.final_memory, Memory)
    assert isinstance(loaded.final_memory.inner, Inner)
    for expected, actual in zip(jt.leaves(states), jt.leaves(loaded)):
        np.testing.assert_array_equal(expected, actual)


def test_writer_selection_keeps_dict_v2_byte_stable_and_uses_v3_for_typed(
    tmp_path: Path,
) -> None:
    legacy = {"array": np.asarray([1, 2], dtype=np.int32), "metadata": {"label": "stable"}}
    data, payload = evaluation_states_container_bytes(legacy)
    assert payload.schema_version == EVALUATION_STATES_CONTAINER_SCHEMA_VERSION
    assert len(data) == 1020
    assert hashlib.sha256(data).hexdigest() == (
        "188607b9d6cc9c6bfce8ca917c7fea36a09f007b5e31cf9122c7d20a1101ac78"
    )
    legacy_ref = store_evaluation_states_artifact(legacy, root=tmp_path, manifest_id="legacy")
    typed_ref = store_evaluation_states_artifact(_states(), root=tmp_path, manifest_id="typed")
    assert legacy_ref.metadata["schema_version"] == EVALUATION_STATES_CONTAINER_SCHEMA_VERSION
    assert typed_ref.metadata["schema_version"] == EVALUATION_STATES_CONTAINER_SCHEMA_VERSION_V3


def test_v3_requires_exact_caller_structure() -> None:
    data, _ = evaluation_states_container_bytes_v3(_states())
    with pytest.raises(EvaluationStatesContainerError, match="caller-supplied structure"):
        load_evaluation_states_container_bytes(data)

    class Renamed(NamedTuple):
        values: object
        commands: object
        feedback: object
        final_memory: Memory
    renamed = Renamed(*_states())
    with pytest.raises(EvaluationStatesContainerError, match="leaf paths"):
        load_evaluation_states_container_bytes(data, structure=jt.structure(renamed))
    with pytest.raises(EvaluationStatesContainerError, match="leaf count"):
        load_evaluation_states_container_bytes(data, structure=jt.structure((np.asarray(1),)))


def test_v3_rejects_path_colliding_wrong_structure_by_fingerprint() -> None:
    data, payload = evaluation_states_container_bytes_v3({0: np.asarray([1])})

    assert type(payload).model_fields["structure_fingerprint"].is_required()
    with pytest.raises(EvaluationStatesContainerError, match="structure fingerprint"):
        load_evaluation_states_container_bytes(
            data,
            structure=jt.structure([np.asarray([1])]),
        )


def test_v3_rejects_empty_subtree_placement_collision_by_fingerprint() -> None:
    class Payload(NamedTuple):
        a: object

    value = np.asarray([1])
    written = Payload(a=[[value, []]])
    impostor = Payload(a=[[value], []])
    data, _ = evaluation_states_container_bytes_v3(written)

    with pytest.raises(EvaluationStatesContainerError, match="structure fingerprint"):
        load_evaluation_states_container_bytes(data, structure=jt.structure(impostor))

    loaded = load_evaluation_states_container_bytes(data, structure=jt.structure(written))
    assert jt.structure(loaded) == jt.structure(written)
    np.testing.assert_array_equal(loaded.a[0][0], value)


def test_v3_structure_fingerprint_is_hash_seed_independent() -> None:
    script = """
import jax.tree_util as jtu
import numpy as np
from feedbax.contracts.evaluation_states import evaluation_states_container_bytes_v3
class Node:
    def __init__(self, value): self.value = value
jtu.register_pytree_node(
    Node,
    lambda node: ((node.value,), frozenset({'a', 'b'})),
    lambda _, leaves: Node(leaves[0]),
)
_, payload = evaluation_states_container_bytes_v3(Node(np.asarray([1])))
print(payload.structure_fingerprint)
"""
    fingerprints = {
        subprocess.check_output(
            [sys.executable, "-c", script],
            env={**os.environ, "PYTHONHASHSEED": seed},
            text=True,
        ).strip()
        for seed in ("1", "2")
    }
    assert len(fingerprints) == 1


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (lambda p: p["leaves"][0].__setitem__("dtype", "float64"), "metadata mismatch"),
        (lambda p: p["leaves"][0].__setitem__("shape", [99]), "metadata mismatch"),
        (lambda p: p["leaves"][0].__setitem__("sha256", "0" * 64), "digest mismatch"),
    ],
)
def test_v3_rejects_array_record_tamper(mutation, message: str) -> None:
    data, _ = evaluation_states_container_bytes_v3(_states())
    tampered = _rewrite_container(data, mutation)
    with pytest.raises(EvaluationStatesContainerError, match=message):
        load_evaluation_states_container_bytes(tampered, structure=jt.structure(_states()))


def test_v3_rejects_metadata_digest_unknown_member_version_and_truncation() -> None:
    states = Trajectory(*_states()[:3], Memory("step", Inner(_states().final_memory.inner.hidden)))
    data, _ = evaluation_states_container_bytes_v3(states)
    structure = jt.structure(states)
    bad_metadata = _rewrite_container(
        data, lambda p: p["leaves"][3].__setitem__("sha256", "0" * 64)
    )
    with pytest.raises(EvaluationStatesContainerError, match="digest mismatch"):
        load_evaluation_states_container_bytes(bad_metadata, structure=structure)
    with pytest.raises(EvaluationStatesContainerError, match="unknown members"):
        load_evaluation_states_container_bytes(
            _rewrite_container(data, lambda _: None, extra=True), structure=structure
        )
    future = _rewrite_container(
        data,
        lambda p: p.__setitem__(
            "schema_version", "feedbax.manifest.evaluation_states_container.v4"
        ),
    )
    with pytest.raises(UnsupportedSpecVersion):
        load_evaluation_states_container_bytes(future, structure=structure)
    with pytest.raises(zipfile.BadZipFile):
        load_evaluation_states_container_bytes(data[:20], structure=structure)


@pytest.mark.parametrize("failure", ["media", "size", "sha", "storage"])
def test_authenticated_v3_wrapper_rejects_evidence_mismatch(
    tmp_path: Path,
    failure: str,
) -> None:
    data, _ = evaluation_states_container_bytes_v3(_states())
    artifact = store_bytes_artifact(
        data,
        root=tmp_path,
        role=EVALUATION_STATES_ARTIFACT_ROLE,
        logical_name="typed.states.npz",
        media_type=EVALUATION_STATES_MEDIA_TYPE,
        metadata={
            "schema_id": "feedbax.manifest.evaluation_states_container",
            "schema_version": EVALUATION_STATES_CONTAINER_SCHEMA_VERSION_V3,
            "storage_backend": EVALUATION_STATES_STORAGE_BACKEND_V3,
            "manifest_id": "evaluation",
        },
    )
    if failure == "media":
        artifact = artifact.model_copy(update={"media_type": "application/octet-stream"})
    elif failure == "size":
        artifact = artifact.model_copy(update={"size_bytes": artifact.size_bytes + 1})
    elif failure == "sha":
        artifact = artifact.model_copy(update={"sha256": "0" * 64, "artifact_id": f"artifact://sha256/{'0' * 64}"})
    else:
        artifact = artifact.model_copy(
            update={"metadata": {**artifact.metadata, "storage_backend": "npz.v2"}}
        )
    with pytest.raises(EvaluationStatesContainerError):
        load_authenticated_evaluation_states_artifact(
            artifact,
            root=tmp_path,
            manifest_id="evaluation",
            structure=jt.structure(_states()),
        )
