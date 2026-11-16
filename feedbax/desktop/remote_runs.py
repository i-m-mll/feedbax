"""Compose the installed Studio service with its configured RunPod Jobs venue."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Annotated, Literal

from pydantic import Field, field_validator

from feedbax._distribution_provenance import load_and_verify_provenance
from feedbax.contracts.base import StrictModel
from feedbax.contracts.execution_target import SessionTargetProfile
from feedbax.desktop.attestation import DesktopApproval
from feedbax.desktop.control import _read_private_json
from feedbax.desktop.release import AppPaths, Release
from feedbax.host.io import private_directory
from feedbax.orchestration.repo_snapshot import SealedRepoSnapshot, verify_repo_snapshot
from feedbax.orchestration.runpod import SubprocessRunPodTransport, load_runpod_api_key
from feedbax.orchestration.runpod_cutoff import NativeRunPodCutoff
from feedbax.orchestration.runpod_session import RunPodEnvironment, RunPodSessionProvider
from feedbax.orchestration.runpod_session_provider import RunPodAcquisition, RunPodResources
from feedbax.orchestration.session import Positive, SessionDriver
from feedbax.web.services.remote_runs import StudioRemoteRuns, StudioVenue


class DesktopRunPod(StrictModel):
    schema_id: Literal["feedbax.desktop.runpod.v1"]
    profiles: Annotated[tuple[SessionTargetProfile, ...], Field(min_length=1)]
    environment: RunPodEnvironment
    authorization_seconds: Positive
    snapshot_root: Path
    runpod_config: Path
    ssh_identity: Path
    uv_binary: Path | None = None

    @field_validator("snapshot_root", "runpod_config", "ssh_identity", "uv_binary")
    @classmethod
    def _absolute(cls, path: Path | None) -> Path | None:
        if path is not None and (not path.is_absolute() or ".." in path.parts):
            raise ValueError("Studio RunPod paths must be absolute.")
        return path


def load_remote_runs(paths: AppPaths, release: Release) -> StudioRemoteRuns | None:
    try:
        document = _read_private_json(paths.control / "runpod.json")
    except FileNotFoundError:
        return None
    config = DesktopRunPod.model_validate(document)
    environment = config.environment
    record = environment.snapshot
    if record.dirty or record.commit != release.document["source_commit"]:
        raise ValueError("The RunPod snapshot must match the installed Studio source commit.")
    verify_repo_snapshot(
        config.snapshot_root, content_sha256=record.content_sha256, file_count=record.file_count,
    )
    provenance, revision = load_and_verify_provenance(config.snapshot_root / "feedbax")
    if revision != record.commit or hashlib.sha256(provenance).hexdigest() != record.distribution_provenance_sha256:
        raise ValueError("The RunPod snapshot provenance differs from the installed release.")
    if (config.uv_binary is None) != (environment.uv_sha256 is None):
        raise ValueError("Studio RunPod requires the declared Linux uv binary for staged uv.")

    for profile in config.profiles:
        if profile.driver != RunPodSessionProvider.target:
            raise ValueError("Studio RunPod profiles must use the RunPod session driver.")
        RunPodResources.model_validate(profile.resources)
    api_key = load_runpod_api_key(config.runpod_config)
    if not api_key or not config.ssh_identity.is_file():
        raise ValueError("The configured RunPod API or SSH identity is unavailable.")

    root = private_directory(paths.state / "jobs" / "runpod")
    approval = DesktopApproval(root / "studio-approval.json", release.resource("launcher"))
    cutoff = NativeRunPodCutoff(environment.workspace, environment.bootstrap_python)
    acquisition = RunPodAcquisition(
        root / "provider", SubprocessRunPodTransport(), api_key=api_key,
        ssh_key_path=config.ssh_identity, running_cutoff=cutoff,
    )
    snapshot = SealedRepoSnapshot("feedbax", config.snapshot_root, config.snapshot_root, record)
    provider = RunPodSessionProvider(acquisition, environment, snapshot, uv_binary=config.uv_binary)
    driver = SessionDriver(root / "controller", provider, verify_authorization=approval.verify)
    venue = StudioVenue(
        driver, environment.digest, provider.jobs, config.authorization_seconds,
        quote_rate=provider.review_rate,
    )
    return StudioRemoteRuns(config.profiles, (venue,), attest=approval)
