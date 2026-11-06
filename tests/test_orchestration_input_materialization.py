import hashlib
import shlex
import shutil
import subprocess
import sys
import tracemalloc
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import pytest

import feedbax.orchestration.bundle as contracts
import feedbax.orchestration.drivers.local as local_driver
import feedbax.orchestration.drivers.runpod as runpod_driver
from feedbax.contracts.artifact_custody import (
    ContentBinding,
    ImmutableArtifactBlobProviderSpec,
)
from feedbax.contracts.spec_storage import (
    canonicalize_immutable_input_identities,
    training_run_execution_hash,
)
from feedbax.orchestration.drivers.local import LocalDriverError, LocalOrchestrationDriver
from feedbax.orchestration.drivers.runpod import (
    CommandResult,
    RunPodDriverConfig,
    RunPodDriverError,
    RunPodOrchestrationDriver,
)
from feedbax.orchestration.input_materialization import (
    InputMaterializationError,
    InputProviderRootBinding,
    acquire_content_projection_lease,
    compile_content_input_projection,
    materialize_content_input_projection,
    staged_execution_bindings_for_bundle,
)
from feedbax.orchestration.stages import run_preflight_checks
from feedbax.orchestration.state import RowState, RunSetState
from feedbax.persistence.artifact_custody import ImmutableArtifactBlobProvider
from feedbax.persistence.content_retention import ContentRetentionError
from feedbax.training.checkpoint_custody import produce_checkpoint_custody_archive
from tests.test_checkpoint_custody import _resolver_parent_ref, _write_resolver_checkpoint
from tests.test_orchestration_core import _bundle
from tests.test_runpod_orchestration_driver import FakeRunPodTransport, _git_seal_ready


def _authenticated_bundle(tmp_path: Path) -> tuple[contracts.RunBundle, Path]:
    parent, root = (
        _resolver_parent_ref(checkpoint := _write_resolver_checkpoint(tmp_path / "checkpoint")),
        tmp_path / "provider",
    )
    artifact = produce_checkpoint_custody_archive(
        parent, allowed_root=checkpoint.root, artifact_provider=ImmutableArtifactBlobProvider(root)
    ).artifact_ref
    identity = contracts.ImmutableInputIdentity(
        role="checkpoint",
        kind="checkpoint-custody-archive",
        identifier=parent.id,
        digest={"value": artifact.sha256},
    )
    custody = contracts.InputCustodySource(
        target_role="checkpoint",
        provider=ImmutableArtifactBlobProviderSpec(),
        provider_binding="checkpoint.inputs",
        artifact=contracts.ImmutableInputArtifactRef(
            **artifact.model_dump(
                include={"artifact_id", "sha256", "size_bytes", "media_type", "storage_backend"}
            )
        ),
        format=contracts.InputFormatIdentity(
            format_id=contracts.CHECKPOINT_CUSTODY_ARCHIVE_FORMAT_ID,
            format_version=contracts.CHECKPOINT_CUSTODY_ARCHIVE_FORMAT_VERSION,
            media_type=contracts.CHECKPOINT_CUSTODY_ARCHIVE_MEDIA_TYPE,
        ),
        materializer=contracts.CheckpointCustodyArchiveMaterializer(
            expected_parent_ref=parent,
            expected_transaction_root_sha256=checkpoint.manifest.content_integrity_digest.transaction_root_sha256,
        ),
    )
    resolved, bundle = (
        contracts.ResolvedAssemblyInput(identity=identity, custody=custody),
        _bundle(tmp_path / "bundle"),
    )
    payload, canonical = (
        bundle.model_dump(mode="json"),
        canonicalize_immutable_input_identities([identity]),
    )
    execution = payload["rows"][0]["execution"]
    execution["immutable_inputs"] = canonical
    execution["execution_capsule"]["execution_hash"] = training_run_execution_hash(
        execution["resolved_snapshot"]["root_hash"], canonical
    )
    payload["resolved_inputs"] = [resolved.model_dump(mode="json")]
    return contracts.RunBundle.model_validate(payload), root


def _state(bundle: contracts.RunBundle, attempts: int = 0) -> RunSetState:
    return RunSetState(
        run_set_id=bundle.run_set_id,
        environment_fingerprint="fingerprint",
        stages={"STAGE_INPUTS": {"attempts": attempts}},
        rows={row.row_id: RowState() for row in bundle.rows},
    )


def _local(root: Path) -> LocalOrchestrationDriver:
    return LocalOrchestrationDriver(
        freeze_lines=[],
        input_provider_bindings=[InputProviderRootBinding("checkpoint.inputs", root)],
    )


class _LeaseManager:
    def __init__(self) -> None:
        self.acquired: list[tuple[str, tuple[str, ...]]] = []
        self.renewed: list[str] = []
        self.released: list[str] = []

    def acquire_lease(self, lease_id: str, roots: tuple[object, ...]) -> None:
        self.acquired.append((lease_id, tuple(root.tree_id for root in roots)))

    def renew_lease(self, lease_id: str) -> None:
        self.renewed.append(lease_id)

    def release_lease(self, lease_id: str) -> None:
        self.released.append(lease_id)


def _content_bundle(
    tmp_path: Path, *, checkpoint_bytes: int | None = None
) -> tuple[contracts.RunBundle, Path, int]:
    store_root = tmp_path / "content-store"
    provider = ImmutableArtifactBlobProvider(store_root)
    selected = tmp_path / "selected"
    selected.mkdir()
    checkpoint = selected / "checkpoint.json"
    if checkpoint_bytes is None:
        checkpoint.write_bytes(b"checkpoint")
    else:
        with checkpoint.open("wb") as stream:
            stream.truncate(checkpoint_bytes)
    (selected / "manifest.json").write_text("manifest", encoding="utf-8")
    tree = provider.ingest_tree(selected)
    provider.store_object(b"unrelated optimizer state")
    provider.store_object(b"unrelated cache artifact")
    bindings = [
        ContentBinding(binding_name="paired_trial_bank", root_kind="manifest-store", tree=tree),
        ContentBinding(binding_name="training", root_kind="manifest-store", tree=tree),
    ]
    bundle = _bundle(tmp_path / "bundle").model_copy(update={"content_bindings": bindings})
    unique_bytes = tree.descriptor.size_bytes + sum(
        member.object.size_bytes for member in provider.get_tree(tree).files
    )
    return bundle, store_root, unique_bytes


class _LocalStagingTransport(FakeRunPodTransport):
    """Run the remote staging commands in a temporary workspace without SSH."""

    interrupt_content_upload = False

    def ssh(self, command: str) -> CommandResult:
        self.ssh_commands.append(command)
        command = command.replace("uv run --no-sync python ", f"{shlex.quote(sys.executable)} ", 1)
        result = subprocess.run(
            ["/bin/sh", "-c", command], capture_output=True, text=True, timeout=20
        )
        return CommandResult(result.returncode, result.stdout, result.stderr)

    def rsync(
        self,
        source: str,
        target: str,
        *,
        delete: bool = False,
        excludes: tuple[str, ...] = (),
        timeout_seconds: float | None = None,
    ) -> CommandResult:
        result = super().rsync(
            source, target, delete=delete, excludes=excludes, timeout_seconds=timeout_seconds
        )
        if self.interrupt_content_upload and Path(target).parent.name == "content-upload":
            return CommandResult(1, "", "interrupted")
        if result.returncode == 0:
            if Path(source).is_dir():
                shutil.copytree(
                    source, target, dirs_exist_ok=True, ignore=shutil.ignore_patterns(*excludes)
                )
            else:
                shutil.copy2(source, target)
        return result


@pytest.fixture
def runpod_staging_config(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> RunPodDriverConfig:
    if sys.platform == "darwin":
        # RunPod's final publication command uses Linux renameat2 directly.
        publish = """
import os
import sys
from pathlib import Path
from feedbax._secure_fs import rename_no_replace
source, destination = map(Path, sys.argv[1:])
source_fd = os.open(source.parent, os.O_RDONLY | os.O_DIRECTORY)
destination_fd = os.open(destination.parent, os.O_RDONLY | os.O_DIRECTORY)
try:
    rename_no_replace(
        source.name,
        destination.name,
        source_dir_fd=source_fd,
        destination_dir_fd=destination_fd,
        error_factory=RuntimeError,
        context="test staging publication",
    )
finally:
    os.close(source_fd)
    os.close(destination_fd)
"""
        monkeypatch.setattr(
            runpod_driver,
            "build_atomic_directory_publish_command",
            lambda source, target: shlex.join([sys.executable, "-c", publish, source, target]),
        )
    workspace = tmp_path / "remote"
    pinned_repo = workspace / "pinned-repo"
    old_package = pinned_repo / "feedbax"
    old_package.mkdir(parents=True)
    (old_package / "__init__.py").write_text("", encoding="utf-8")
    return RunPodDriverConfig(
        volume_mount=str(workspace),
        remote_run_root=str(workspace / "runs"),
        remote_repos={"feedbax": str(pinned_repo)},
    )


def test_content_aliases_share_one_local_projection_and_exclude_unselected_objects(
    tmp_path: Path,
) -> None:
    bundle, store_root, unique_bytes = _content_bundle(tmp_path)
    driver = LocalOrchestrationDriver(
        freeze_lines=[], content_store_root=store_root, content_lease_seconds=3600
    )

    result = driver.stage_inputs(bundle, _state(bundle))

    content = [item for item in result["inputs"] if item["target_role"].startswith("content/")]
    assert len(content) == 2
    assert content[0]["destination"] == content[1]["destination"]
    tree_root = Path(content[0]["destination"])
    assert sorted(path.name for path in tree_root.iterdir()) == [
        "checkpoint.json",
        "manifest.json",
    ]
    selected_files = [
        path for path in (bundle.run_set_dir / "inputs/content").rglob("*") if path.is_file()
    ]
    assert sum(path.stat().st_size for path in selected_files) == unique_bytes
    ImmutableArtifactBlobProvider(store_root).release_lease(bundle.run_set_id)
    runtime = staged_execution_bindings_for_bundle(
        bundle, inputs_root=bundle.run_set_dir / "inputs"
    )
    assert len(runtime.manifest_root_bindings) == 2
    assert runtime.manifest_root_bindings[0].root == runtime.manifest_root_bindings[1].root


def test_runpod_content_views_are_isolated_and_reuse_uploaded_objects(
    tmp_path: Path, runpod_staging_config: RunPodDriverConfig
) -> None:
    bundle, store_root, unique_bytes = _content_bundle(tmp_path)
    transport = _LocalStagingTransport()
    driver = RunPodOrchestrationDriver(
        config=runpod_staging_config,
        transport=transport,
        content_store_root=store_root,
        content_lease_seconds=3600,
    )

    def stage(run_set_id: str) -> tuple[dict[str, Any], Path]:
        run = bundle.model_copy(update={"run_set_id": run_set_id})
        projection = driver.stage_inputs(run, _state(run, 1))["content_projection"]
        assert projection["bindings"][0]["target"] == projection["bindings"][1]["target"]
        target = Path(projection["bindings"][0]["target"])
        inputs = Path(runpod_staging_config.remote_run_root) / run_set_id / "inputs"
        runtime = staged_execution_bindings_for_bundle(run, inputs_root=inputs)
        assert target == runtime.manifest_root_bindings[0].root
        return projection, target

    first, first_view = stage(bundle.run_set_id)
    second, second_view = stage("2026-01-02-feedface")
    assert first["uploaded_bytes"] == unique_bytes
    assert first["transfer_bytes"] == unique_bytes + first["metadata_bytes"]
    assert first["controller_helper_bytes"] > 0
    assert first["uploaded_object_count"] == first["unique_object_count"]
    assert second["uploaded_bytes"] == second["uploaded_object_count"] == 0
    descriptor = bundle.content_bindings[0].tree.descriptor
    first_descriptor = first_view.parents[1] / "descriptors" / descriptor.sha256
    second_descriptor = second_view.parents[1] / "descriptors" / descriptor.sha256
    original_descriptor = second_descriptor.read_bytes()

    (first_view / "checkpoint.json").write_bytes(b"experiment")
    first_descriptor.write_bytes(b"changed descriptor")

    assert (second_view / "checkpoint.json").read_bytes() == b"checkpoint"
    assert second_descriptor.read_bytes() == original_descriptor
    third, third_view = stage("2026-01-02-abcddcba")
    assert third["uploaded_bytes"] == third["uploaded_object_count"] == 0
    assert (third_view / "checkpoint.json").read_bytes() == b"checkpoint"
    object_uploads = [
        Path(target).name
        for _, target, _, _ in transport.rsync_calls
        if Path(target).parent.name == "content-upload"
    ]
    assert len(object_uploads) == len(set(object_uploads)) == first["unique_object_count"]


def test_runpod_rejects_conflicting_remote_digest_before_upload(
    tmp_path: Path, runpod_staging_config: RunPodDriverConfig
) -> None:
    bundle, store_root, _ = _content_bundle(tmp_path)
    transport = _LocalStagingTransport()
    digest = bundle.content_bindings[0].tree.descriptor.sha256
    cache = Path(runpod_staging_config.volume_mount) / "feedbax-content-cache/v2/objects/sha256"
    conflicting = cache / digest[:2] / digest
    conflicting.parent.mkdir(parents=True)
    conflicting.write_bytes(b"conflicting bytes")
    driver = RunPodOrchestrationDriver(
        config=runpod_staging_config,
        transport=transport,
        content_store_root=store_root,
        content_lease_seconds=3600,
    )

    with pytest.raises(RunPodDriverError, match="remote content cache conflicts"):
        driver.stage_inputs(bundle, _state(bundle))
    assert [Path(target).name for _, target, _, _ in transport.rsync_calls] == [
        "content-staging.zip"
    ]
    with pytest.raises(ContentRetentionError, match="active lease is unavailable"):
        ImmutableArtifactBlobProvider(store_root).renew_lease(
            bundle.run_set_id,
            expires_at=datetime.now(timezone.utc) + timedelta(hours=2),
        )


def test_content_source_mutation_fails_before_materialization(tmp_path: Path) -> None:
    bundle, store_root, _ = _content_bundle(tmp_path)
    provider = ImmutableArtifactBlobProvider(store_root)
    member = provider.get_tree(bundle.content_bindings[0].tree).files[0].object
    canonical = store_root / "artifacts" / "sha256" / member.sha256[:2] / member.sha256
    canonical.chmod(0o600)
    canonical.write_bytes(b"x" * member.size_bytes)
    with pytest.raises(Exception, match="SHA-256 mismatch"):
        compile_content_input_projection(bundle.content_bindings, provider)


def test_local_content_views_are_isolated_from_each_other_and_custody(tmp_path: Path) -> None:
    bundle, store_root, _ = _content_bundle(tmp_path)
    provider = ImmutableArtifactBlobProvider(store_root)
    projection = compile_content_input_projection(bundle.content_bindings, provider)

    def materialize(name: str) -> tuple[Path, Path]:
        inputs = tmp_path / name / "inputs"
        views = materialize_content_input_projection(
            projection,
            provider,
            destination_root=inputs / "content" / "trees",
        )
        return inputs, views[0].destination

    first_inputs, first = materialize("first")
    second_inputs, second = materialize("second")
    descriptor_ref = bundle.content_bindings[0].tree.descriptor
    first_descriptor = first_inputs / "content" / "descriptors" / descriptor_ref.sha256
    second_descriptor = second_inputs / "content" / "descriptors" / descriptor_ref.sha256
    original_descriptor = second_descriptor.read_bytes()

    (first / "checkpoint.json").write_bytes(b"experiment")
    first_descriptor.write_bytes(b"changed descriptor")

    assert (second / "checkpoint.json").read_bytes() == b"checkpoint"
    assert second_descriptor.read_bytes() == original_descriptor
    assert provider.get_object(descriptor_ref) == original_descriptor
    staged_execution_bindings_for_bundle(bundle, inputs_root=second_inputs)

    third_inputs, third = materialize("third")
    assert (third / "checkpoint.json").read_bytes() == b"checkpoint"
    staged_execution_bindings_for_bundle(bundle, inputs_root=third_inputs)


@pytest.mark.parametrize("invalid_alias", [0, 1])
def test_content_aliases_reject_conflicting_descriptor_size(
    tmp_path: Path, invalid_alias: int
) -> None:
    bundle, store_root, _ = _content_bundle(tmp_path)
    provider = ImmutableArtifactBlobProvider(store_root)
    inputs = tmp_path / "view" / "inputs"
    projection = compile_content_input_projection(bundle.content_bindings, provider)
    materialize_content_input_projection(
        projection, provider, destination_root=inputs / "content" / "trees"
    )

    bindings = list(bundle.content_bindings)
    binding = bindings[invalid_alias]
    descriptor = binding.tree.descriptor.model_copy(
        update={"size_bytes": binding.tree.descriptor.size_bytes + 1}
    )
    bindings[invalid_alias] = binding.model_copy(
        update={"tree": binding.tree.model_copy(update={"descriptor": descriptor})}
    )
    conflicting_bundle = bundle.model_copy(update={"content_bindings": bindings})

    with pytest.raises(InputMaterializationError, match="metadata conflicts"):
        compile_content_input_projection(bindings, provider)
    with pytest.raises(InputMaterializationError, match="metadata conflicts"):
        staged_execution_bindings_for_bundle(conflicting_bundle, inputs_root=inputs)


def test_large_content_member_verification_has_bounded_memory(tmp_path: Path) -> None:
    member_size = 16 * 1024 * 1024
    bundle, store_root, _ = _content_bundle(tmp_path, checkpoint_bytes=member_size)
    driver = LocalOrchestrationDriver(
        freeze_lines=[], content_store_root=store_root, content_lease_seconds=3600
    )
    inputs = bundle.run_set_dir / "inputs"

    tracemalloc.start()
    try:
        driver.stage_inputs(bundle, _state(bundle))
        runtime = staged_execution_bindings_for_bundle(bundle, inputs_root=inputs)
        _, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
    assert peak < member_size // 2
    member = runtime.manifest_root_bindings[0].root / "checkpoint.json"
    with member.open("r+b") as stream:
        stream.seek(member_size - 1)
        stream.write(b"x")
    with pytest.raises(InputMaterializationError, match="SHA-256 mismatch"):
        staged_execution_bindings_for_bundle(bundle, inputs_root=inputs)


def test_concurrent_identical_local_projections_converge(tmp_path: Path) -> None:
    bundle, store_root, _ = _content_bundle(tmp_path)
    provider = ImmutableArtifactBlobProvider(store_root)
    projection = compile_content_input_projection(bundle.content_bindings, provider)
    destination = tmp_path / "shared-view" / "trees"

    with ThreadPoolExecutor(max_workers=2) as pool:
        results = list(
            pool.map(
                lambda _index: materialize_content_input_projection(
                    projection, provider, destination_root=destination
                ),
                range(2),
            )
        )

    assert results[0][0].destination == results[1][0].destination
    assert results[0][0].destination == results[0][1].destination


def test_interrupted_remote_object_upload_never_publishes_inputs(
    tmp_path: Path, runpod_staging_config: RunPodDriverConfig
) -> None:
    bundle, store_root, _ = _content_bundle(tmp_path)
    transport = _LocalStagingTransport()
    transport.interrupt_content_upload = True
    driver = RunPodOrchestrationDriver(
        config=runpod_staging_config,
        transport=transport,
        content_store_root=store_root,
        content_lease_seconds=3600,
    )

    with pytest.raises(RunPodDriverError, match="upload immutable content object.*interrupted"):
        driver.stage_inputs(bundle, _state(bundle))
    assert not (Path(runpod_staging_config.remote_run_root) / bundle.run_set_id / "inputs").exists()
    with pytest.raises(ContentRetentionError, match="active lease is unavailable"):
        ImmutableArtifactBlobProvider(store_root).renew_lease(
            bundle.run_set_id,
            expires_at=datetime.now(timezone.utc) + timedelta(hours=2),
        )


def test_missing_explicit_content_lease_policy_fails_before_remote_contact(tmp_path: Path) -> None:
    bundle, store_root, _ = _content_bundle(tmp_path)
    transport = FakeRunPodTransport()
    driver = RunPodOrchestrationDriver(
        transport=transport,
        content_store_root=store_root,
    )

    with pytest.raises(Exception, match="explicit lease seconds"):
        driver.stage_inputs(bundle, _state(bundle))
    assert transport.operations == []


def test_terminal_teardown_releases_content_lease(tmp_path: Path) -> None:
    bundle, store_root, _ = _content_bundle(tmp_path)
    provider = ImmutableArtifactBlobProvider(store_root)
    projection = compile_content_input_projection(bundle.content_bindings, provider)
    acquire_content_projection_lease(
        projection,
        provider,
        lease_id=bundle.run_set_id,
        owner=f"runpod:{bundle.run_set_id}",
        expires_at=datetime.now(timezone.utc) + timedelta(hours=1),
    )
    driver = RunPodOrchestrationDriver(
        transport=FakeRunPodTransport(),
        content_store_root=store_root,
        content_lease_seconds=3600,
    )

    result = driver.teardown(bundle, _state(bundle))

    assert result["content_lease"] == "released"
    with pytest.raises(ContentRetentionError, match="active lease is unavailable"):
        provider.renew_lease(
            bundle.run_set_id,
            expires_at=datetime.now(timezone.utc) + timedelta(hours=2),
        )


def test_publication_collision_preserves_attempt_and_retry_publishes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    bundle, root = _authenticated_bundle(tmp_path)
    row = bundle.rows[0]
    launch = row.launch.model_copy(
        update={"payload_routing": {"kind": "registered-execution-payload"}}
    )
    bundle = bundle.model_copy(update={"rows": [row.model_copy(update={"launch": launch})]})
    original_copy = local_driver.shutil.copy2
    monkeypatch.setattr(
        local_driver.shutil,
        "copy2",
        lambda source, target: (
            (bundle.run_set_dir / "inputs").mkdir(),
            original_copy(source, target),
        )[1],
    )
    with pytest.raises(FileExistsError, match="won publication race"):
        _local(root).stage_inputs(bundle, _state(bundle, 1))
    residue = bundle.run_set_dir / ".stage-attempts/stage-inputs-1/inputs/checkpoint"
    assert residue.is_dir() and not (bundle.run_set_dir / "inputs/checkpoint").exists()
    (bundle.run_set_dir / "inputs").rmdir()
    monkeypatch.setattr(local_driver.shutil, "copy2", original_copy)
    assert (
        _local(root).stage_inputs(bundle, _state(bundle, 2))["input_count"] == 1
        and residue.is_dir()
        and (bundle.run_set_dir / "inputs/checkpoint").is_dir()
    )


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
    assert checks["runpod-credentials"].observed["outcome"] == ("skipped-due-to-dependency")
    assert not transport.operations


def test_materialization_rejects_wrong_transaction_authority(tmp_path: Path) -> None:
    bundle, root = _authenticated_bundle(tmp_path)
    authority = (resolved := bundle.resolved_inputs[0]).custody.materializer.model_copy(
        update={"expected_transaction_root_sha256": "f" * 64}
    )
    custody = resolved.custody.model_copy(update={"materializer": authority})
    bundle = bundle.model_copy(
        update={"resolved_inputs": [resolved.model_copy(update={"custody": custody})]}
    )
    with pytest.raises(LocalDriverError, match="archive document identity|authority"):
        _local(root).stage_inputs(bundle, _state(bundle))


def test_missing_artifact_preflight_fails_before_runpod_queries(tmp_path: Path) -> None:
    bundle, root = _authenticated_bundle(tmp_path)
    resolved = bundle.resolved_inputs[0]
    digest = "f" * 64
    artifact = resolved.custody.artifact.model_copy(
        update={"artifact_id": f"artifact://sha256/{digest}", "sha256": digest}
    )
    custody = resolved.custody.model_copy(update={"artifact": artifact})
    identity = resolved.identity.model_copy(
        update={"digest": resolved.identity.digest.model_copy(update={"value": digest})}
    )
    bundle = bundle.model_copy(
        update={
            "resolved_inputs": [
                resolved.model_copy(update={"identity": identity, "custody": custody})
            ]
        }
    )
    transport = FakeRunPodTransport()
    driver = RunPodOrchestrationDriver(
        transport=transport,
        input_provider_bindings=[InputProviderRootBinding("checkpoint.inputs", root)],
    )
    checks = {check.name: check for check in driver.preflight_checks(bundle)}
    assert checks["input-provider-bindings"].status == "fail"
    assert transport.operations == []


def test_preflight_rejects_invalid_target_role(tmp_path: Path) -> None:
    bundle, _ = _authenticated_bundle(tmp_path)
    resolved = bundle.resolved_inputs[0]
    custody = resolved.custody.model_copy(update={"target_role": "../checkpoint"})
    bundle = bundle.model_copy(
        update={"resolved_inputs": [resolved.model_copy(update={"custody": custody})]}
    )
    check = {
        item.name: item
        for item in run_preflight_checks(
            bundle,
            LocalOrchestrationDriver.realized_capabilities,
        )
    }["input-custody-authority"]
    assert check.status == "fail" and "target_role" in (check.detail or "")


def test_runpod_transfers_and_verifies_every_materialized_file(tmp_path: Path) -> None:
    bundle, root = _authenticated_bundle(tmp_path)
    transport = FakeRunPodTransport()
    driver = RunPodOrchestrationDriver(
        config=RunPodDriverConfig(ssh_host="198.51.100.10", ssh_port=2222),
        transport=transport,
        input_provider_bindings=[InputProviderRootBinding("checkpoint.inputs", root)],
    )
    transfer = driver.stage_inputs(bundle, _state(bundle))["inputs"][0]
    assert transfer["target"].endswith("/inputs/checkpoint") and transport.rsync_calls[0][2]
    assert ".stage-attempts/stage-inputs-0" in transfer["source"]
    assert (
        len([cmd for cmd in transport.ssh_commands if "sha256sum -c" in cmd])
        >= transfer["file_count"]
    )
