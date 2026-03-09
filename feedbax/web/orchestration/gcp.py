"""GCP instance orchestration via gcloud CLI subprocess."""
from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass
from enum import Enum
from typing import Optional

from feedbax.web.orchestration.startup_script import make_startup_script


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
        auth_token: Optional shared secret for the worker's auth middleware.
        ts_auth_key: Optional Tailscale auth key for network provisioning.
    """

    project: str
    zone: str
    machine_type: str = "n1-standard-4"
    image_family: str = "debian-11"
    image_project: str = "debian-cloud"
    preemptible: bool = True
    worker_port: int = 8765
    auth_token: Optional[str] = None
    ts_auth_key: Optional[str] = None
    feedbax_install_cmd: str = (
        "pip install 'git+https://github.com/mlll-io/feedbax.git@develop'"
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
        RuntimeError: If gcloud exits with a non-zero return code, with the
            stderr output included in the message.
    """
    proc = await asyncio.create_subprocess_exec(
        "gcloud",
        *args,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    stdout, stderr = await proc.communicate()
    if proc.returncode != 0:
        raise RuntimeError(
            f"gcloud {' '.join(args)} failed (exit {proc.returncode}):\n"
            + stderr.decode(errors="replace")
        )
    return json.loads(stdout.decode())


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
    machine_type = machine_type_url.rsplit("/", 1)[-1] if "/" in machine_type_url else machine_type_url

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

    The instance's startup script installs feedbax, optionally connects to
    Tailscale, and starts the worker process.

    Args:
        config: Instance configuration (project, zone, machine type, etc.).
        instance_name: Name to give the instance.

    Returns:
        An :class:`InstanceInfo` describing the newly created instance.

    Raises:
        RuntimeError: If ``gcloud`` fails.
    """
    metadata_parts = [f"startup-script={make_startup_script(config.feedbax_install_cmd)}"]
    if config.ts_auth_key:
        metadata_parts.append(f"TS_AUTH_KEY={config.ts_auth_key}")
    metadata_parts.append(f"WORKER_PORT={config.worker_port}")
    if config.auth_token:
        metadata_parts.append(f"AUTH_TOKEN={config.auth_token}")
    metadata_str = ",".join(metadata_parts)

    args = [
        "compute", "instances", "create", instance_name,
        f"--project={config.project}",
        f"--zone={config.zone}",
        f"--machine-type={config.machine_type}",
        f"--image-family={config.image_family}",
        f"--image-project={config.image_project}",
        f"--metadata={metadata_str}",
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
        "compute", "instances", "describe", name,
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
        "compute", "instances", "delete", name,
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
        "compute", "instances", "list",
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
