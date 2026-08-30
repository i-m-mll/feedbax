from types import SimpleNamespace

import numpy as np
import pytest

from feedbax.analysis import (
    EvaluationChannelEvidenceError,
    resolve_authenticated_evaluation_channels,
)
from feedbax.analysis.execution_context import (
    EMPTY_STAGED_EXECUTION_CONTEXT,
    StagedParentExecutionLocation,
    with_staged_parent_execution_locations,
)
from feedbax.analysis.manifest_inputs import authenticated_manifest_ref
from feedbax.contracts.evaluation_states import store_evaluation_states_artifact
from feedbax.contracts.manifest import (
    ArtifactRef,
    EvaluationRunManifest,
    ParentRef,
    SpecPayload,
    StagedEvaluationPrerequisite,
    write_manifest,
)


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

    def _resolve_evaluation_states(
        self,
        _parent,
        *,
        prerequisite_artifact_provider,
        validate_staged_prerequisite,
    ):
        assert prerequisite_artifact_provider == "provider"
        assert validate_staged_prerequisite is True
        return SimpleNamespace(
            states=self.states,
            manifest_input=SimpleNamespace(manifest=self.manifest),
        )


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


def test_authenticated_channels_reuse_trial_bank_material_authority(tmp_path):
    array = np.arange(6, dtype=np.float64).reshape(2, 3)
    artifact = store_evaluation_states_artifact(
        {"channels": {"noise": array}},
        root=tmp_path,
        manifest_id="feedbax-evaluation-run:trial-bank",
    )
    artifact = artifact.model_copy(update={"uri": artifact.metadata["relative_path"]})
    manifest = EvaluationRunManifest(
        id="feedbax-evaluation-run:trial-bank",
        status="completed",
        evaluation_spec=SpecPayload(
            kind="EvaluationRunSpec",
            schema_id="feedbax.spec.evaluation_run",
            schema_version="feedbax.spec.evaluation_run.v1",
            inline={
                "schema_version": "feedbax.spec.evaluation_run.v1",
                "evaluation_type": "test",
                "training_run_ids": [],
                "inputs": [],
                "params": {},
            },
        ),
        artifacts=[artifact],
        metadata={"channels": [_record(array)]},
    )
    path = write_manifest(manifest, root=tmp_path, index=False)
    executable_parent = authenticated_manifest_ref(manifest, path, "evaluation_run")
    staged_parent = executable_parent.model_copy(update={"role": "paired_trial_bank"})
    context = with_staged_parent_execution_locations(
        EMPTY_STAGED_EXECUTION_CONTEXT,
        [
            StagedParentExecutionLocation(
                parent=staged_parent,
                root=tmp_path,
                execution_uri=path.relative_to(tmp_path).as_posix(),
            )
        ],
    )

    resolved = resolve_authenticated_evaluation_channels(
        StagedEvaluationPrerequisite(parent=executable_parent),
        execution_context=context,
    )

    assert resolved.channels["noise"] is resolved.states["channels"]["noise"]
    assert resolved.manifest_sha256 == executable_parent.metadata["manifest_sha256"]
    assert resolved.states_sha256 == artifact.sha256


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
