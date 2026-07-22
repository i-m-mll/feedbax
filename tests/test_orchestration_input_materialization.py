import hashlib
from pathlib import Path

import pytest

import feedbax.orchestration.bundle as contracts
import feedbax.orchestration.drivers.local as local_driver
from feedbax.contracts.artifact_custody import ImmutableArtifactBlobProviderSpec
from feedbax.contracts.spec_storage import canonicalize_immutable_input_identities, training_run_execution_hash
from feedbax.orchestration.drivers.local import LocalDriverError, LocalOrchestrationDriver
from feedbax.orchestration.drivers.runpod import RunPodDriverConfig, RunPodOrchestrationDriver
from feedbax.orchestration.input_materialization import InputProviderRootBinding
from feedbax.orchestration.stages import run_preflight_checks
from feedbax.orchestration.state import RowState, RunSetState
from feedbax.persistence.artifact_custody import ImmutableArtifactBlobProvider
from feedbax.training.checkpoint_custody import produce_checkpoint_custody_archive
from tests.test_checkpoint_custody import _resolver_parent_ref, _write_resolver_checkpoint
from tests.test_orchestration_core import _bundle
from tests.test_runpod_orchestration_driver import FakeRunPodTransport, _git_seal_ready


def _authenticated_bundle(tmp_path: Path) -> tuple[contracts.RunBundle, Path]:
    parent, root = _resolver_parent_ref(
        checkpoint := _write_resolver_checkpoint(tmp_path / "checkpoint")), tmp_path / "provider"
    artifact = produce_checkpoint_custody_archive(parent, allowed_root=checkpoint.root,
        artifact_provider=ImmutableArtifactBlobProvider(root)).artifact_ref
    identity = contracts.ImmutableInputIdentity(role="checkpoint", kind="checkpoint-custody-archive",
        identifier=parent.id, digest={"value": artifact.sha256})
    custody = contracts.InputCustodySource(
        target_role="checkpoint", provider=ImmutableArtifactBlobProviderSpec(),
        provider_binding="checkpoint.inputs",
        artifact=contracts.ImmutableInputArtifactRef(**artifact.model_dump(include={"artifact_id", "sha256",
            "size_bytes", "media_type", "storage_backend"})),
        format=contracts.InputFormatIdentity(format_id=contracts.CHECKPOINT_CUSTODY_ARCHIVE_FORMAT_ID,
            format_version=contracts.CHECKPOINT_CUSTODY_ARCHIVE_FORMAT_VERSION,
            media_type=contracts.CHECKPOINT_CUSTODY_ARCHIVE_MEDIA_TYPE),
        materializer=contracts.CheckpointCustodyArchiveMaterializer(expected_parent_ref=parent,
            expected_transaction_root_sha256=
                checkpoint.manifest.content_integrity_digest.transaction_root_sha256),
    )
    resolved, bundle = contracts.ResolvedAssemblyInput(identity=identity, custody=custody), _bundle(tmp_path / "bundle")
    payload, canonical = bundle.model_dump(mode="json"), canonicalize_immutable_input_identities([identity])
    execution = payload["rows"][0]["execution"]
    execution["immutable_inputs"] = canonical
    execution["execution_capsule"]["execution_hash"] = training_run_execution_hash(
        execution["resolved_snapshot"]["root_hash"], canonical)
    payload["resolved_inputs"] = [resolved.model_dump(mode="json")]
    return contracts.RunBundle.model_validate(payload), root


def _state(bundle: contracts.RunBundle, attempts: int = 0) -> RunSetState:
    return RunSetState(run_set_id=bundle.run_set_id, environment_fingerprint="fingerprint",
        stages={"STAGE_INPUTS": {"attempts": attempts}},
        rows={row.row_id: RowState() for row in bundle.rows})


def _local(root: Path) -> LocalOrchestrationDriver:
    return LocalOrchestrationDriver(freeze_lines=[], input_provider_bindings=
                                    [InputProviderRootBinding("checkpoint.inputs", root)])


def test_publication_collision_preserves_attempt_and_retry_publishes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    bundle, root = _authenticated_bundle(tmp_path)
    row = bundle.rows[0]
    launch = row.launch.model_copy(update={"payload_routing":
                                          {"kind": "registered-execution-payload"}})
    bundle = bundle.model_copy(update={"rows": [row.model_copy(update={"launch": launch})]})
    original_copy = local_driver.shutil.copy2
    monkeypatch.setattr(local_driver.shutil, "copy2", lambda source, target:
        ((bundle.run_set_dir / "inputs").mkdir(), original_copy(source, target))[1])
    with pytest.raises(FileExistsError, match="won publication race"):
        _local(root).stage_inputs(bundle, _state(bundle, 1))
    residue = bundle.run_set_dir / ".stage-attempts/stage-inputs-1/inputs/checkpoint"
    assert residue.is_dir() and not (bundle.run_set_dir / "inputs/checkpoint").exists()
    (bundle.run_set_dir / "inputs").rmdir()
    monkeypatch.setattr(local_driver.shutil, "copy2", original_copy)
    assert _local(root).stage_inputs(bundle, _state(bundle, 2))["input_count"] == 1 and residue.is_dir() and (bundle.run_set_dir / "inputs/checkpoint").is_dir()


def test_existing_real_directory_rejects_before_materialization(tmp_path: Path) -> None:
    bundle, root = _authenticated_bundle(tmp_path)
    (bundle.run_set_dir / "inputs/checkpoint").mkdir(parents=True)
    with pytest.raises(LocalDriverError, match="destination already exists"):
        _local(root).stage_inputs(bundle, _state(bundle))


@pytest.mark.parametrize("target_exists", [True, False])
def test_existing_materialization_rejects_symlink_target(
    tmp_path: Path, target_exists: bool
) -> None:
    bundle, root = _authenticated_bundle(tmp_path)
    target, elsewhere = bundle.run_set_dir / "inputs/checkpoint", tmp_path / "elsewhere"
    target.parent.mkdir(parents=True)
    if target_exists:
        elsewhere.mkdir()
    target.symlink_to(elsewhere, target_is_directory=True)
    with pytest.raises(LocalDriverError, match="destination already exists"):
        _local(root).stage_inputs(bundle, _state(bundle))


def test_runtime_binding_preflight_fails_before_runpod_queries(tmp_path: Path) -> None:
    bundle, _ = _authenticated_bundle(tmp_path)
    repo_root = tmp_path / "repo"
    lockfile = repo_root / "uv.lock"
    lockfile.parent.mkdir()
    lockfile.write_text("version = 1\n", encoding="utf-8")
    _git_seal_ready(repo_root, "uv.lock")
    environment = bundle.environment.model_copy(
        update={"lockfile_hashes": {"uv.lock": hashlib.sha256(lockfile.read_bytes()).hexdigest()}}
    )
    bundle = bundle.model_copy(update={"environment": environment})
    transport = FakeRunPodTransport()
    driver = RunPodOrchestrationDriver(
        config=RunPodDriverConfig(local_repos={"feedbax": repo_root}),
        transport=transport,
    )
    checks = {check.name: check for check in driver.preflight_checks(bundle)}
    assert checks["input-provider-bindings"].status == "fail"
    assert checks["continuation-schedule-consistency"].status == "pass"
    assert checks["continuation-schedule-consistency"].observed == "no-continuations"
    assert checks["runpod-credentials"].observed["outcome"] == (
        "skipped-due-to-dependency"
    )
    assert not transport.operations


def test_materialization_rejects_wrong_transaction_authority(tmp_path: Path) -> None:
    bundle, root = _authenticated_bundle(tmp_path)
    authority = (resolved := bundle.resolved_inputs[0]).custody.materializer.model_copy(update={"expected_transaction_root_sha256": "f" * 64})
    custody = resolved.custody.model_copy(update={"materializer": authority})
    bundle = bundle.model_copy(update={"resolved_inputs": [resolved.model_copy(update={"custody": custody})]})
    with pytest.raises(LocalDriverError, match="archive document identity|authority"):
        _local(root).stage_inputs(bundle, _state(bundle))


def test_missing_artifact_preflight_fails_before_runpod_queries(tmp_path: Path) -> None:
    bundle, root = _authenticated_bundle(tmp_path)
    resolved = bundle.resolved_inputs[0]
    digest = "f" * 64
    artifact = resolved.custody.artifact.model_copy(update={"artifact_id": f"artifact://sha256/{digest}", "sha256": digest})
    custody = resolved.custody.model_copy(update={"artifact": artifact})
    identity = resolved.identity.model_copy(update={"digest": resolved.identity.digest.model_copy(update={"value": digest})})
    bundle = bundle.model_copy(update={"resolved_inputs": [resolved.model_copy(update={"identity": identity, "custody": custody})]})
    transport = FakeRunPodTransport()
    driver = RunPodOrchestrationDriver(
        transport=transport, input_provider_bindings=[InputProviderRootBinding("checkpoint.inputs", root)])
    checks = {check.name: check for check in driver.preflight_checks(bundle)}
    assert checks["input-provider-bindings"].status == "fail"
    assert transport.operations == []


def test_preflight_rejects_invalid_target_role(tmp_path: Path) -> None:
    bundle, _ = _authenticated_bundle(tmp_path)
    resolved = bundle.resolved_inputs[0]
    custody = resolved.custody.model_copy(update={"target_role": "../checkpoint"})
    bundle = bundle.model_copy(update={"resolved_inputs": [resolved.model_copy(update={"custody": custody})]})
    check = {item.name: item for item in run_preflight_checks(bundle)}["input-custody-authority"]
    assert check.status == "fail" and "target_role" in (check.detail or "")


def test_runpod_transfers_and_verifies_every_materialized_file(tmp_path: Path) -> None:
    bundle, root = _authenticated_bundle(tmp_path)
    transport = FakeRunPodTransport()
    driver = RunPodOrchestrationDriver(
        config=RunPodDriverConfig(ssh_host="198.51.100.10", ssh_port=2222), transport=transport,
        input_provider_bindings=[InputProviderRootBinding("checkpoint.inputs", root)])
    transfer = driver.stage_inputs(bundle, _state(bundle))["inputs"][0]
    assert transfer["target"].endswith("/inputs/checkpoint") and transport.rsync_calls[0][2]
    assert ".stage-attempts/stage-inputs-0" in transfer["source"]
    assert len([cmd for cmd in transport.ssh_commands if "sha256sum -c" in cmd]) >= transfer["file_count"]
