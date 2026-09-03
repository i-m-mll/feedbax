"""GCP instance orchestration via gcloud CLI subprocess."""

from __future__ import annotations

import asyncio
import json
import os
import tempfile
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Optional

from feedbax.web.orchestration.startup_script import (
    FeedbaxInstallSpec,
    make_startup_script,
)


_GCLOUD_TERMINATION_GRACE_SECONDS = 5.0


class InstanceStatus(str, Enum):
    CREATING = "CREATING"
    RUNNING = "RUNNING"
    STAGING = "STAGING"
    STOPPING = "STOPPING"
    TERMINATED = "TERMINATED"
    PREEMPTED = "PREEMPTED"
    UNKNOWN = "UNKNOWN"


# Map GCP status strings to our InstanceStatus enum.
_GCP_STATUS_MAP: dict[str, InstanceStatus] = {
    "RUNNING": InstanceStatus.RUNNING,
    "STAGING": InstanceStatus.STAGING,
    "STOPPING": InstanceStatus.STOPPING,
    "TERMINATED": InstanceStatus.TERMINATED,
    "SUSPENDED": InstanceStatus.TERMINATED,
    "PROVISIONING": InstanceStatus.CREATING,
    "REPAIRING": InstanceStatus.UNKNOWN,
}


@dataclass
class InstanceConfig:
    """Configuration for a GCP compute instance that will run the Feedbax worker.

    Args:
        project: GCP project ID.
        zone: GCP zone, e.g. ``"us-central1-a"``.
        machine_type: GCP machine type, e.g. ``"n1-standard-4"`` or
            ``"ct5lp-hightpu-4t"`` for TPU.
        image_family: Boot disk image family.
        image_project: GCP project owning the image family.
        preemptible: Whether to use a preemptible (spot) instance.
        worker_port: Port the Feedbax worker will bind to.
        auth_token: Required shared secret for the worker's auth middleware.
        ts_auth_key: Unsupported legacy Tailscale auth key input.
        install_spec: Structured Feedbax install request for the startup script.
    """

    project: str
    zone: str
    machine_type: str = "n1-standard-4"
    image_family: str = "debian-11"
    image_project: str = "debian-cloud"
    preemptible: bool = True
    worker_port: int = 8765
    auth_token: str = ""
    ts_auth_key: Optional[str] = None
    install_spec: FeedbaxInstallSpec = field(default_factory=FeedbaxInstallSpec)

    def __post_init__(self) -> None:
        if not self.auth_token:
            raise ValueError("GCP workers require an explicit worker credential")
        if self.ts_auth_key:
            raise ValueError(
                "GCP Tailscale bootstrap is unsupported because it cannot keep credentials "
                "out of process arguments"
            )


@dataclass
class InstanceInfo:
    """Runtime information about a GCP compute instance.

    Args:
        name: Instance name.
        status: Current lifecycle status.
        internal_ip: Internal (VPC) IP address, if available.
        external_ip: External (public) IP address, if available.
        tailscale_ip: Tailscale IP address, set after the worker advertises it.
        zone: GCP zone the instance is running in.
        machine_type: GCP machine type of the instance.
    """

    name: str
    status: InstanceStatus
    internal_ip: Optional[str] = None
    external_ip: Optional[str] = None
    tailscale_ip: Optional[str] = None
    zone: str = ""
    machine_type: str = ""


# ---------------------------------------------------------------------------
# Internal helper
# ---------------------------------------------------------------------------


async def _run_gcloud(*args: str) -> dict | list:
    """Run a gcloud command and return parsed JSON output.

    Args:
        *args: Arguments passed to ``gcloud`` (after the binary name).

    Returns:
        Parsed JSON object (dict or list) from gcloud stdout.

    Raises:
        RuntimeError: If gcloud exits with a non-zero return code. Provider
            output is kept out of the exception because it may contain secrets.
    """
    proc = await asyncio.create_subprocess_exec(
        "gcloud",
        *args,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    try:
        stdout, _stderr = await proc.communicate()
    except BaseException:
        cleanup_task = asyncio.create_task(_reap_gcloud_process(proc))
        while not cleanup_task.done():
            try:
                await asyncio.shield(cleanup_task)
            except asyncio.CancelledError:
                continue
            except BaseException:
                break
        if cleanup_task.done():
            try:
                cleanup_task.result()
            except BaseException:
                pass
        raise
    if proc.returncode != 0:
        raise RuntimeError(f"gcloud command failed (exit {proc.returncode}); inspect provider logs")
    return json.loads(stdout.decode())


async def _reap_gcloud_process(proc: asyncio.subprocess.Process) -> None:
    """Boundedly stop one gcloud child while draining its output pipes."""
    if proc.returncode is None:
        try:
            proc.terminate()
        except ProcessLookupError:
            pass

    drain_task = asyncio.create_task(proc.communicate())
    try:
        await asyncio.wait_for(
            asyncio.shield(drain_task),
            timeout=_GCLOUD_TERMINATION_GRACE_SECONDS,
        )
        return
    except TimeoutError:
        try:
            proc.kill()
        except ProcessLookupError:
            pass

    try:
        await asyncio.wait_for(
            asyncio.shield(drain_task),
            timeout=_GCLOUD_TERMINATION_GRACE_SECONDS,
        )
    finally:
        if not drain_task.done():
            drain_task.cancel()
            try:
                await drain_task
            except BaseException:
                pass


def _parse_instance(raw: dict) -> InstanceInfo:
    """Extract an :class:`InstanceInfo` from a raw gcloud JSON instance dict.

    Args:
        raw: A single instance dict as returned by ``gcloud compute instances
            describe`` or one element of ``gcloud compute instances list``.

    Returns:
        A populated :class:`InstanceInfo`.
    """
    gcp_status = raw.get("status", "UNKNOWN").upper()
    status = _GCP_STATUS_MAP.get(gcp_status, InstanceStatus.UNKNOWN)

    # Extract IPs from the networkInterfaces list.
    internal_ip: Optional[str] = None
    external_ip: Optional[str] = None
    for iface in raw.get("networkInterfaces", []):
        if internal_ip is None:
            internal_ip = iface.get("networkIP")
        for access_cfg in iface.get("accessConfigs", []):
            if external_ip is None:
                external_ip = access_cfg.get("natIP")

    # Extract short machine type and zone from their self-link paths.
    machine_type_url: str = raw.get("machineType", "")
    machine_type = (
        machine_type_url.rsplit("/", 1)[-1] if "/" in machine_type_url else machine_type_url
    )

    zone_url: str = raw.get("zone", "")
    zone = zone_url.rsplit("/", 1)[-1] if "/" in zone_url else zone_url

    return InstanceInfo(
        name=raw.get("name", ""),
        status=status,
        internal_ip=internal_ip,
        external_ip=external_ip,
        zone=zone,
        machine_type=machine_type,
    )


# ---------------------------------------------------------------------------
# Public async API
# ---------------------------------------------------------------------------


async def create_instance(config: InstanceConfig, instance_name: str) -> InstanceInfo:
    """Create a GCP compute instance and return its initial info.

    The instance's startup script installs feedbax and starts the authenticated
    worker process. This low-level helper does not establish a trusted HTTPS
    origin; the Studio adapter therefore refuses to invoke it.

    Args:
        config: Instance configuration (project, zone, machine type, etc.).
        instance_name: Name to give the instance.

    Returns:
        An :class:`InstanceInfo` describing the newly created instance.

    Raises:
        RuntimeError: If ``gcloud`` fails.
    """
    with tempfile.TemporaryDirectory(prefix="feedbax-gcp-metadata-") as directory:
        metadata_dir = Path(directory)
        os.chmod(metadata_dir, 0o700)
        startup_path = metadata_dir / "startup.sh"
        token_path = metadata_dir / "worker-token"
        startup_path.write_text(make_startup_script(config.install_spec), encoding="utf-8")
        token_path.write_text(config.auth_token, encoding="utf-8")
        os.chmod(startup_path, 0o600)
        os.chmod(token_path, 0o600)

        args = [
            "compute",
            "instances",
            "create",
            instance_name,
            f"--project={config.project}",
            f"--zone={config.zone}",
            f"--machine-type={config.machine_type}",
            f"--image-family={config.image_family}",
            f"--image-project={config.image_project}",
            f"--metadata=WORKER_PORT={config.worker_port}",
            f"--metadata-from-file=startup-script={startup_path},AUTH_TOKEN={token_path}",
            "--format=json",
        ]
        if config.preemptible:
            args.append("--preemptible")

        raw = await _run_gcloud(*args)
    # gcloud returns a list when creating an instance.
    instances: list = raw if isinstance(raw, list) else [raw]
    if not instances:
        raise RuntimeError("gcloud create returned empty response")
    return _parse_instance(instances[0])


async def get_instance(project: str, zone: str, name: str) -> InstanceInfo:
    """Describe a single GCP instance and return its current info.

    Args:
        project: GCP project ID.
        zone: GCP zone.
        name: Instance name.

    Returns:
        An :class:`InstanceInfo` with the current status and IPs.

    Raises:
        RuntimeError: If ``gcloud`` fails.
    """
    raw = await _run_gcloud(
        "compute",
        "instances",
        "describe",
        name,
        f"--project={project}",
        f"--zone={zone}",
        "--format=json",
    )
    return _parse_instance(raw)  # type: ignore[arg-type]


async def delete_instance(project: str, zone: str, name: str) -> None:
    """Delete a GCP compute instance.

    Args:
        project: GCP project ID.
        zone: GCP zone.
        name: Instance name.

    Raises:
        RuntimeError: If ``gcloud`` fails.
    """
    await _run_gcloud(
        "compute",
        "instances",
        "delete",
        name,
        f"--project={project}",
        f"--zone={zone}",
        "--quiet",
    )


async def list_instances(project: str, zone: str) -> list[InstanceInfo]:
    """List all Feedbax worker instances in the given project and zone.

    Only instances whose names start with ``"feedbax-worker-"`` are returned.

    Args:
        project: GCP project ID.
        zone: GCP zone.

    Returns:
        A list of :class:`InstanceInfo` objects.

    Raises:
        RuntimeError: If ``gcloud`` fails.
    """
    raw = await _run_gcloud(
        "compute",
        "instances",
        "list",
        f"--project={project}",
        f"--zones={zone}",
        "--format=json",
    )
    instances: list = raw if isinstance(raw, list) else []
    return [
        _parse_instance(item)
        for item in instances
        if item.get("name", "").startswith("feedbax-worker-")
    ]
