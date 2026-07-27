from __future__ import annotations

import os
import shutil
from pathlib import Path

import pytest
from pydantic import ValidationError

import feedbax.orchestration.input_materialization as input_materialization
from feedbax.analysis.execution_context import resolve_staged_execution_context
from feedbax.contracts.artifact_custody import ImmutableArtifactBlobProviderSpec
from feedbax.orchestration.bundle import RunBundle
from feedbax.orchestration.drivers.local import LocalOrchestrationDriver
from feedbax.orchestration.drivers.runpod import (
    RunPodDriverConfig,
    RunPodOrchestrationDriver,
)
from feedbax.orchestration.input_materialization import (
    InputMaterializationError,
    materialize_bundle_inputs,
    preflight_staged_root_bindings,
    staged_execution_bindings_for_bundle,
)
from feedbax.orchestration.staged_root_custody import (
    StagedRootCustody,
    StagedRootCustodyError,
    StagedRootSnapshotBinding,
    StagedRootSourceBinding,
    seal_staged_root,
    verify_staged_root_snapshot,
)
from feedbax.orchestration.state import RowState, RunSetState
from feedbax.persistence.artifact_custody import ImmutableArtifactBlobProvider
from tests.test_orchestration_core import _bundle
from tests.test_runpod_orchestration_driver import FakeRunPodTransport


def _sealed_bundle(
    tmp_path: Path,
) -> tuple[RunBundle, tuple[StagedRootSnapshotBinding, ...]]:
    sources = tmp_path / "sources"
    manifest_root = sources / "manifest"
    artifact_root = sources / "artifacts"
    checkpoint_root = sources / "checkpoints"
    manifest_path = manifest_root / "manifests" / "training_runs" / "run.json"
    manifest_path.parent.mkdir(parents=True)
    manifest_path.write_text('{"id":"run"}\n', encoding="utf-8")
    artifact_root.mkdir(parents=True)
    ImmutableArtifactBlobProvider(artifact_root).store_bytes(
        b"artifact-bytes",
        role="fixture",
        logical_name="fixture",
    )
    checkpoint_path = checkpoint_root / "transactions" / "tx" / "model.eqx"
    checkpoint_path.parent.mkdir(parents=True)
    checkpoint_path.write_bytes(b"checkpoint-bytes")

    source_bindings = (
        StagedRootSourceBinding(
            "artifacts",
            "artifact-provider",
            artifact_root,
            ImmutableArtifactBlobProviderSpec(),
        ),
        StagedRootSourceBinding(
            "checkpoints",
            "checkpoint-custody",
            checkpoint_root,
        ),
        StagedRootSourceBinding("manifests", "manifest-store", manifest_root),
    )
    sealed = tuple(
        seal_staged_root(binding, snapshot_parent=tmp_path / "sealed")
        for binding in source_bindings
    )
    bundle = _bundle(tmp_path / "orchestration").model_copy(
        update={"staged_roots": [item.custody for item in sealed]}
    )
    return bundle, tuple(item.binding for item in sealed)


def _state(bundle: RunBundle, attempts: int = 1) -> RunSetState:
    return RunSetState(
        run_set_id=bundle.run_set_id,
        environment_fingerprint="fingerprint",
        stages={"STAGE_INPUTS": {"attempts": attempts}},
        rows={row.row_id: RowState() for row in bundle.rows},
    )


def _make_writable(path: Path) -> None:
    path.parent.chmod(0o700)
    path.chmod(0o600)


def test_staged_root_contract_round_trip_and_schema_rejection(tmp_path: Path) -> None:
    bundle, bindings = _sealed_bundle(tmp_path)

    payload = bundle.staged_roots[0].model_dump(mode="json")
    assert StagedRootCustody.model_validate(payload) == bundle.staged_roots[0]
    assert bundle == RunBundle.model_validate_json(bundle.model_dump_json())
    assert [record.relative_path for record in bundle.staged_roots[0].files] == sorted(
        record.relative_path for record in bundle.staged_roots[0].files
    )
    assert bundle.staged_roots[0].custody_ref.startswith("staged-root://sha256/")
    assert not preflight_staged_root_bindings(bundle, bindings)[0]

    payload["schema_version"] = "feedbax.orchestration.staged_root_custody.v0"
    with pytest.raises(ValidationError):
        StagedRootCustody.model_validate(payload)


def test_staged_roots_do_not_change_native_training_row_bytes(tmp_path: Path) -> None:
    native = _bundle(tmp_path / "native")
    staged, _bindings = _sealed_bundle(tmp_path / "staged")
    rebound = native.model_copy(update={"staged_roots": staged.staged_roots})

    assert rebound.rows == native.rows
    assert rebound.resolved_inputs == native.resolved_inputs


def test_sealing_rejects_symlinks_and_binding_name_drift(tmp_path: Path) -> None:
    source = tmp_path / "source"
    source.mkdir()
    (source / "real").write_text("bytes", encoding="utf-8")
    (source / "link").symlink_to(source / "real")
    with pytest.raises(StagedRootCustodyError, match="symlink or unsupported"):
        seal_staged_root(
            StagedRootSourceBinding("manifests", "manifest-store", source),
            snapshot_parent=tmp_path / "sealed",
        )

    bundle, bindings = _sealed_bundle(tmp_path / "drift")
    first = bindings[0]
    drifted = (
        StagedRootSnapshotBinding(
            "other",
            first.kind,
            first.root,
            first.expected_root_identity,
        ),
        *bindings[1:],
    )
    failures, _ = preflight_staged_root_bindings(bundle, drifted)
    assert any("missing=" in failure and "unexpected=" in failure for failure in failures)


def test_snapshot_drift_fails_local_preflight_before_provision(tmp_path: Path) -> None:
    bundle, bindings = _sealed_bundle(tmp_path)
    mutated = Path(bindings[0].root) / bundle.staged_roots[0].files[0].relative_path
    _make_writable(mutated)
    mutated.write_bytes(b"mutated")

    driver = LocalOrchestrationDriver(
        freeze_lines=[],
        staged_root_bindings=bindings,
    )
    checks = driver.preflight_checks(bundle)

    assert checks[0].status == "fail"
    assert "differs from custody manifest" in (checks[0].detail or "")
    assert not bundle.run_set_dir.exists()


def test_byte_identical_snapshot_replacement_is_rejected(tmp_path: Path) -> None:
    bundle, bindings = _sealed_bundle(tmp_path)
    replaced = Path(bindings[0].root)
    backup = replaced.with_name(replaced.name + ".original")
    replaced.rename(backup)
    shutil.copytree(backup, replaced)

    failures, _ = preflight_staged_root_bindings(bundle, bindings)

    assert any("binding was replaced" in failure for failure in failures)


def test_local_materialization_projects_exact_public_context(tmp_path: Path) -> None:
    bundle, bindings = _sealed_bundle(tmp_path)
    staged = materialize_bundle_inputs(
        bundle,
        destination_root=bundle.run_set_dir / ".stage-attempts" / "focused",
        staged_root_bindings=bindings,
    )
    inputs_root = bundle.run_set_dir / ".stage-attempts" / "focused" / "inputs"

    assert len(staged) == 3
    projected = staged_execution_bindings_for_bundle(bundle, inputs_root=inputs_root)
    context = resolve_staged_execution_context(
        projected.descriptor,
        artifact_provider_bindings=projected.artifact_provider_bindings,
        manifest_root_bindings=projected.manifest_root_bindings,
        checkpoint_custody_bindings=projected.checkpoint_custody_bindings,
    )

    assert tuple(context.opened_artifact_providers) == ("artifacts",)
    assert tuple(context.checkpoint_custody_roots) == ("checkpoints",)
    assert tuple(context.manifest_roots) == ("manifests",)
    assert context.artifact_provider("artifacts").root == (
        inputs_root / "staged-roots" / "artifact-provider" / "artifacts"
    )


def test_staged_root_group_cleans_every_partial_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    bundle, bindings = _sealed_bundle(tmp_path)
    original = input_materialization.materialize_staged_root_snapshot
    calls = 0

    def fail_second(*args, **kwargs):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise StagedRootCustodyError("injected materialization failure")
        return original(*args, **kwargs)

    monkeypatch.setattr(
        input_materialization,
        "materialize_staged_root_snapshot",
        fail_second,
    )
    destination_root = bundle.run_set_dir / ".stage-attempts" / "failure"
    with pytest.raises(InputMaterializationError, match="injected materialization failure"):
        materialize_bundle_inputs(
            bundle,
            destination_root=destination_root,
            staged_root_bindings=bindings,
        )

    inputs_root = destination_root / "inputs"
    assert not (inputs_root / "staged-roots").exists()
    assert not list(inputs_root.glob(".staged-roots-*"))


def test_runpod_stage_inputs_transports_only_sealed_materialization(
    tmp_path: Path,
) -> None:
    bundle, bindings = _sealed_bundle(tmp_path)
    transport = FakeRunPodTransport()
    driver = RunPodOrchestrationDriver(
        config=RunPodDriverConfig(remote_run_root="/remote/runs"),
        transport=transport,
        staged_root_bindings=bindings,
    )

    result = driver.stage_inputs(bundle, _state(bundle))

    assert result["input_count"] == 3
    assert len(transport.rsync_calls) == 1
    source, target, delete, _exclude = transport.rsync_calls[0]
    assert source.endswith("/inputs/")
    assert target.endswith("/inputs/")
    assert delete is True
    assert not transport.runpodctl_calls


def test_unexpected_member_is_rejected_without_replacing_custody(
    tmp_path: Path,
) -> None:
    bundle, bindings = _sealed_bundle(tmp_path)
    root = Path(bindings[-1].root)
    root.chmod(0o700)
    unexpected = root / "unexpected.json"
    unexpected.write_text("{}", encoding="utf-8")

    with pytest.raises(StagedRootCustodyError, match="differs from custody manifest"):
        verify_staged_root_snapshot(bundle.staged_roots[-1], root)
    assert os.path.samefile(root, bindings[-1].root)
