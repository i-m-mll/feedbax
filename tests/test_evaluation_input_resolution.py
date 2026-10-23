from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import pytest

import feedbax.analysis.evaluation_inputs as evaluation_inputs_module
from feedbax.analysis import (
    EvaluationInputAmbiguityError,
    EvaluationInputCardinalityError,
    EvaluationInputIntegrityError,
    EvaluationInputManifestError,
    EvaluationInputPathError,
    EvaluationInputReferenceError,
    ResolvedEvaluationInput,
    resolve_evaluation_inputs,
)
from feedbax.contracts.base import ParentRef
from feedbax.contracts.manifest import (
    EvaluationRunSpec,
    TrainingRunManifest,
    spec_payload,
)


MANIFEST_ID = "feedbax-training-run:resolved-input"
MANIFEST_URI = "manifests/training_runs/feedbax-training-run_resolved-input.json"


def _training_manifest() -> TrainingRunManifest:
    return TrainingRunManifest(
        id=MANIFEST_ID,
        graph_spec=ParentRef(kind="GraphSpecManifest", id="graph:governed"),
        training_spec=spec_payload("TrainingRunSpec", {"n_batches": 4}),
        task_spec=spec_payload("TaskSpec", {"task_type": "reach"}),
        checkpoint_custody=[
            ParentRef(
                kind="TrainingCheckpointTransactionManifest",
                id="checkpoint-transaction:exact",
                role="checkpoint_transaction",
                uri="manifests/checkpoints/exact.json",
            )
        ],
    )


def _write_manifest(root: Path, manifest: TrainingRunManifest | None = None) -> tuple[Path, bytes]:
    manifest = manifest or _training_manifest()
    path = root / MANIFEST_URI
    path.parent.mkdir(parents=True, exist_ok=True)
    raw_bytes = (manifest.model_dump_json(indent=2, exclude_none=True) + "\n").encode()
    path.write_bytes(raw_bytes)
    return path, raw_bytes


def _ref(**updates: object) -> ParentRef:
    payload: dict[str, object] = {
        "kind": "TrainingRunManifest",
        "id": MANIFEST_ID,
        "role": "training_run",
        "uri": MANIFEST_URI,
    }
    payload.update(updates)
    return ParentRef.model_validate(payload)


def _spec(*refs: ParentRef) -> EvaluationRunSpec:
    return EvaluationRunSpec(evaluation_type="tests.resolve", inputs=list(refs))


def test_resolve_evaluation_inputs_returns_exact_verified_training_manifest(
    tmp_path: Path,
) -> None:
    path, raw_bytes = _write_manifest(tmp_path)
    digest = hashlib.sha256(raw_bytes).hexdigest()
    ref = _ref(metadata={"manifest_sha256": digest})

    resolved = resolve_evaluation_inputs(_spec(ref), manifest_root=tmp_path)

    assert len(resolved) == 1
    item = resolved[0]
    assert isinstance(item, ResolvedEvaluationInput)
    assert item.ref is ref
    assert item.manifest == _training_manifest()
    assert item.manifest.checkpoint_custody == _training_manifest().checkpoint_custody
    assert item.path == path.resolve()
    assert item.reference == MANIFEST_URI
    assert item.sha256 == digest
    assert item.kind == "TrainingRunManifest"
    assert item.id == MANIFEST_ID
    assert not (tmp_path / "index").exists()


def test_resolve_evaluation_inputs_rejects_absent_input(tmp_path: Path) -> None:
    with pytest.raises(EvaluationInputCardinalityError, match="got zero"):
        resolve_evaluation_inputs(_spec(), manifest_root=tmp_path)


def test_resolve_evaluation_inputs_rejects_multiple_inputs(tmp_path: Path) -> None:
    refs = (_ref(), _ref(id="feedbax-training-run:other", uri="other.json"))
    with pytest.raises(EvaluationInputCardinalityError, match="got 2"):
        resolve_evaluation_inputs(_spec(*refs), manifest_root=tmp_path)


def test_resolve_evaluation_inputs_rejects_duplicate_declaration(tmp_path: Path) -> None:
    ref = _ref()
    with pytest.raises(EvaluationInputReferenceError, match="duplicate ParentRef"):
        resolve_evaluation_inputs(_spec(ref, ref.model_copy(deep=True)), manifest_root=tmp_path)


@pytest.mark.parametrize(
    ("updates", "match"),
    [
        ({"role": "analysis_run"}, "role must be 'training_run'"),
        ({"kind": "EvaluationRunManifest"}, "kind must be 'TrainingRunManifest'"),
    ],
)
def test_resolve_evaluation_inputs_rejects_wrong_ref_contract(
    tmp_path: Path,
    updates: dict[str, object],
    match: str,
) -> None:
    with pytest.raises(EvaluationInputReferenceError, match=match):
        resolve_evaluation_inputs(_spec(_ref(**updates)), manifest_root=tmp_path)


def test_resolve_evaluation_inputs_rejects_missing_manifest(tmp_path: Path) -> None:
    with pytest.raises(EvaluationInputReferenceError, match="does not exist"):
        resolve_evaluation_inputs(_spec(_ref()), manifest_root=tmp_path)


def test_resolve_evaluation_inputs_rejects_duplicate_on_disk_id(tmp_path: Path) -> None:
    _write_manifest(tmp_path)
    duplicate = tmp_path / "manifests" / "imported" / "duplicate.json"
    duplicate.parent.mkdir(parents=True)
    duplicate.write_text(_training_manifest().model_dump_json(), encoding="utf-8")

    with pytest.raises(EvaluationInputAmbiguityError, match="is duplicated"):
        resolve_evaluation_inputs(_spec(_ref()), manifest_root=tmp_path)


@pytest.mark.parametrize(
    "uri",
    [
        "https://example.test/manifest.json",
        "file:///tmp/manifest.json",
        "/tmp/manifest.json",
        "../manifest.json",
        "manifests/%2e%2e/manifest.json",
        "manifests\\training_runs\\manifest.json",
    ],
)
def test_resolve_evaluation_inputs_rejects_unsupported_or_escaped_uri(
    tmp_path: Path,
    uri: str,
) -> None:
    with pytest.raises(EvaluationInputPathError):
        resolve_evaluation_inputs(_spec(_ref(uri=uri)), manifest_root=tmp_path)


def test_resolve_evaluation_inputs_rejects_symlink_escape(tmp_path: Path) -> None:
    outside = tmp_path.parent / f"{tmp_path.name}-outside.json"
    outside.write_text(_training_manifest().model_dump_json(), encoding="utf-8")
    link = tmp_path / "manifests" / "training_runs" / "escaped.json"
    link.parent.mkdir(parents=True)
    link.symlink_to(outside)

    with pytest.raises(EvaluationInputPathError, match="symlink or non-directory"):
        resolve_evaluation_inputs(
            _spec(_ref(uri="manifests/training_runs/escaped.json")),
            manifest_root=tmp_path,
        )


@pytest.mark.parametrize("swap_target", ["directory", "file"])
def test_resolve_evaluation_inputs_rejects_path_swap_at_secure_open(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    swap_target: str,
) -> None:
    path, _ = _write_manifest(tmp_path)
    outside_dir = tmp_path.parent / f"{tmp_path.name}-outside-{swap_target}"
    outside_dir.mkdir()
    outside_path = outside_dir / path.name
    outside_path.write_text(_training_manifest().model_dump_json(), encoding="utf-8")
    real_open = os.open
    swapped = False

    def swap_before_open(
        target: str | bytes | os.PathLike[str] | os.PathLike[bytes],
        flags: int,
        mode: int = 0o777,
        *,
        dir_fd: int | None = None,
    ) -> int:
        nonlocal swapped
        target_name = os.fsdecode(target)
        trigger = "training_runs" if swap_target == "directory" else path.name
        if not swapped and dir_fd is not None and target_name == trigger:
            swapped = True
            if swap_target == "directory":
                original = path.parent
                original.rename(original.with_name("training_runs-held"))
                original.symlink_to(outside_dir, target_is_directory=True)
            else:
                path.rename(path.with_suffix(".held"))
                path.symlink_to(outside_path)
        return real_open(target, flags, mode, dir_fd=dir_fd)

    monkeypatch.setattr(evaluation_inputs_module.os, "open", swap_before_open)

    with pytest.raises(EvaluationInputPathError, match="symlink or non-directory"):
        resolve_evaluation_inputs(_spec(_ref()), manifest_root=tmp_path)
    assert swapped


def test_resolve_evaluation_inputs_rejects_non_file_uri(tmp_path: Path) -> None:
    directory = tmp_path / "manifests" / "training_runs" / "directory.json"
    directory.mkdir(parents=True)

    with pytest.raises(EvaluationInputPathError, match="not a regular file"):
        resolve_evaluation_inputs(
            _spec(_ref(uri="manifests/training_runs/directory.json")),
            manifest_root=tmp_path,
        )


@pytest.mark.parametrize("declared", ["A" * 64, "abc", 7])
def test_resolve_evaluation_inputs_rejects_malformed_declared_hash(
    tmp_path: Path,
    declared: object,
) -> None:
    _write_manifest(tmp_path)
    with pytest.raises(EvaluationInputIntegrityError, match="64 lowercase hex"):
        resolve_evaluation_inputs(
            _spec(_ref(metadata={"manifest_sha256": declared})),
            manifest_root=tmp_path,
        )


def test_resolve_evaluation_inputs_rejects_hash_mismatch(tmp_path: Path) -> None:
    _write_manifest(tmp_path)
    with pytest.raises(EvaluationInputIntegrityError, match="does not match"):
        resolve_evaluation_inputs(
            _spec(_ref(metadata={"manifest_sha256": "0" * 64})),
            manifest_root=tmp_path,
        )


@pytest.mark.parametrize(
    ("payload", "match"),
    [
        (b"not-json", "not valid JSON"),
        (json.dumps(["not", "an", "object"]).encode(), "must be an object"),
        (
            json.dumps({"kind": "TrainingRunManifest", "id": MANIFEST_ID, "extra": True}).encode(),
            "not a valid TrainingRunManifest",
        ),
    ],
)
def test_resolve_evaluation_inputs_rejects_invalid_manifest_schema(
    tmp_path: Path,
    payload: bytes,
    match: str,
) -> None:
    path = tmp_path / MANIFEST_URI
    path.parent.mkdir(parents=True)
    path.write_bytes(payload)

    with pytest.raises(EvaluationInputManifestError, match=match):
        resolve_evaluation_inputs(_spec(_ref()), manifest_root=tmp_path)


def test_resolve_evaluation_inputs_rejects_loaded_id_mismatch(tmp_path: Path) -> None:
    manifest = _training_manifest().model_copy(update={"id": "feedbax-training-run:other"})
    _write_manifest(tmp_path, manifest)

    with pytest.raises(EvaluationInputManifestError, match="id does not match"):
        resolve_evaluation_inputs(_spec(_ref()), manifest_root=tmp_path)


def test_resolve_evaluation_inputs_rejects_loaded_kind_mismatch(tmp_path: Path) -> None:
    path, _ = _write_manifest(tmp_path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["kind"] = "EvaluationRunManifest"
    path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(EvaluationInputManifestError, match="valid TrainingRunManifest"):
        resolve_evaluation_inputs(_spec(_ref()), manifest_root=tmp_path)
