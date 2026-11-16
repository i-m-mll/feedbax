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
from feedbax.orchestration.runpod import SubprocessRunPodTransport, read_runpod_api_key
from feedbax.orchestration.runpod_cutoff import NativeRunPodCutoff
from feedbax.orchestration.runpod_session import RunPodEnvironment, RunPodSessionProvider
from feedbax.orchestration.runpod_session_provider import RunPodAcquisition, RunPodResources
from feedbax.orchestration.session import Positive, SessionDriver
from feedbax.web.services.remote_runs import StudioRemoteRuns, StudioVenue


class DesktopRunPod(StrictModel):
    schema_id: Literal["feedbax.desktop.runpod.v1"]
    profiles: Annotated[tuple[SessionTargetProfile, ...], Field(min_length=1)]
    authorization_seconds: Positive
    runpod_config: Path
    ssh_identity: Path

    @field_validator("runpod_config", "ssh_identity")
    @classmethod
    def _absolute(cls, path: Path) -> Path:
        if not path.is_absolute() or ".." in path.parts:
            raise ValueError("Studio RunPod paths must be absolute.")
        return path


def load_remote_runs(paths: AppPaths, release: Release) -> StudioRemoteRuns | None:
    try:
        document = _read_private_json(paths.control / "runpod.json")
    except FileNotFoundError:
        return None
    config = DesktopRunPod.model_validate(document)
    environment = RunPodEnvironment.model_validate_json(release.resource("runpod_environment").read_bytes())
    source = release.resource("runpod_source")
    record = environment.snapshot
    if record.dirty or record.commit != release.document["source_commit"]:
        raise ValueError("The RunPod snapshot must match the installed Studio source commit.")
    verify_repo_snapshot(
        source, content_sha256=record.content_sha256, file_count=record.file_count,
    )
    provenance, revision = load_and_verify_provenance(source / "feedbax")
    if revision != record.commit or hashlib.sha256(provenance).hexdigest() != record.distribution_provenance_sha256:
        raise ValueError("The RunPod snapshot provenance differs from the installed release.")
    for profile in config.profiles:
        if profile.driver != RunPodSessionProvider.target:
            raise ValueError("Studio RunPod profiles must use the RunPod session driver.")
        RunPodResources.model_validate(profile.resources)
    root = private_directory(paths.state / "jobs" / "runpod")
    approval = DesktopApproval(root / "studio-approval.json", release.resource("launcher"))
    api_key = read_runpod_api_key(config.runpod_config)
    if not api_key or not config.ssh_identity.is_file():
        return StudioRemoteRuns(config.profiles, (), attest=approval)
    cutoff = NativeRunPodCutoff(environment.workspace, environment.bootstrap_python)
    acquisition = RunPodAcquisition(
        root / "provider", SubprocessRunPodTransport(), api_key=api_key,
        ssh_key_path=config.ssh_identity, running_cutoff=cutoff,
    )
    snapshot = SealedRepoSnapshot("feedbax", source, source, record)
    provider = RunPodSessionProvider(
        acquisition, environment, snapshot,
        uv_binary=release.resource("runpod_uv"), install_timeout=480,
    )
    driver = SessionDriver(root / "controller", provider, verify_authorization=approval.verify)
    venue = StudioVenue(
        driver, environment.digest, provider.jobs, config.authorization_seconds,
        quote_rate=provider.review_rate,
    )
    return StudioRemoteRuns(config.profiles, (venue,), attest=approval)
