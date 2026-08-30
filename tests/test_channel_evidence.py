from types import SimpleNamespace

import numpy as np
import pytest

from feedbax.analysis import (
    EvaluationChannelEvidenceError,
    resolve_authenticated_evaluation_channels,
)
from feedbax.contracts.manifest import ArtifactRef, ParentRef, StagedEvaluationPrerequisite


class _Context:
    def __init__(self, states, records):
        self.states = states
        self.manifest = SimpleNamespace(
            artifacts=[
                ArtifactRef(
                    role="evaluation_states", logical_name="states", sha256="b" * 64
                )
            ],
            metadata={"channels": records},
        )

    def parent_execution_location(self, _parent):
        return SimpleNamespace(artifact_provider="provider")

    def load_evaluation_states(
        self,
        _parent,
        *,
        prerequisite_artifact_provider,
        validate_staged_prerequisite,
    ):
        assert prerequisite_artifact_provider == "provider"
        assert validate_staged_prerequisite is True
        return self.states

    def resolve_manifest_input(self, _parent):
        return SimpleNamespace(manifest=self.manifest)


def _prerequisite():
    return StagedEvaluationPrerequisite(
        parent=ParentRef(
            kind="EvaluationRunManifest",
            id="evaluation",
            role="evaluation_run",
            metadata={"manifest_sha256": "a" * 64},
        ),
        artifact_provider="provider",
    )


def _record(array, *, name="noise", index=0):
    import hashlib

    return {
        "name": name,
        "index": index,
        "shape": list(array.shape),
        "dtype": array.dtype.str,
        "byte_order": array.dtype.str[0],
        "c_contiguous": True,
        "sha256": hashlib.sha256(array.tobytes(order="C")).hexdigest(),
    }


def test_resolve_authenticated_evaluation_channels_returns_immutable_evidence():
    array = np.arange(6, dtype=np.float64).reshape(2, 3)
    states = {"channels": {"noise": array}, "axis": np.arange(2)}

    resolved = resolve_authenticated_evaluation_channels(
        _prerequisite(), execution_context=_Context(states, [_record(array)])
    )

    assert resolved.states is states
    assert resolved.channels["noise"] is array
    assert resolved.evidence["noise"].sha256 == _record(array)["sha256"]
    assert resolved.manifest_sha256 == "a" * 64
    assert resolved.states_sha256 == "b" * 64
    with pytest.raises(TypeError):
        resolved.evidence["other"] = resolved.evidence["noise"]


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("shape", [3, 2], "shape/dtype/byte-order/contiguity"),
        ("dtype", "<f4", "shape/dtype/byte-order/contiguity"),
        ("byte_order", ">", "shape/dtype/byte-order/contiguity"),
        ("c_contiguous", False, "shape/dtype/byte-order/contiguity"),
        ("sha256", "0" * 64, "SHA-256 mismatch"),
        ("sha256", "BAD", "malformed evidence sha256"),
        ("index", -1, "malformed evidence index"),
    ],
)
def test_resolve_authenticated_evaluation_channels_rejects_record_mismatch(
    field, value, message
):
    array = np.arange(6, dtype=np.float64).reshape(2, 3)
    record = _record(array)
    record[field] = value

    with pytest.raises(EvaluationChannelEvidenceError, match=message):
        resolve_authenticated_evaluation_channels(
            _prerequisite(),
            execution_context=_Context({"channels": {"noise": array}}, [record]),
        )


@pytest.mark.parametrize(
    ("indexes", "expected", "actual"),
    [([0, 0], 1, 0), ([1, 0], 0, 1)],
)
def test_resolve_authenticated_evaluation_channels_rejects_noncanonical_indexes(
    indexes, expected, actual
):
    channels = {
        "noise": np.arange(3, dtype=np.float64),
        "signal": np.arange(3, dtype=np.float64),
    }
    records = [
        _record(channels[name], name=name, index=index)
        for name, index in zip(channels, indexes, strict=True)
    ]

    with pytest.raises(
        EvaluationChannelEvidenceError,
        match=rf"channel .* expected={expected}, actual={actual}",
    ):
        resolve_authenticated_evaluation_channels(
            _prerequisite(), execution_context=_Context({"channels": channels}, records)
        )


@pytest.mark.parametrize(
    ("states", "records", "message"),
    [
        ({}, [], "'channels' mapping"),
        ({"channels": {"noise": np.ones(1)}}, None, "must be a sequence"),
        ({"channels": {"noise": np.ones(1)}}, [], "channel keys disagree"),
        ({"channels": {}}, [{"name": "noise"}], "missing or is not an array"),
    ],
)
def test_resolve_authenticated_evaluation_channels_rejects_missing_evidence(
    states, records, message
):
    with pytest.raises(EvaluationChannelEvidenceError, match=message):
        resolve_authenticated_evaluation_channels(
            _prerequisite(), execution_context=_Context(states, records)
        )
