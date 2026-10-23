"""Worker startup script injected as GCP instance metadata."""

from __future__ import annotations

import re
import shlex
from dataclasses import dataclass, field
from typing import Any


INSTALL_SPEC_SCHEMA_VERSION = "feedbax.orchestration.install.v1"
DEFAULT_FEEDBAX_REPOSITORY = "https://github.com/mlll-io/feedbax.git"
DEFAULT_FEEDBAX_REF = "develop"

_SAFE_REF = re.compile(r"^[A-Za-z0-9._/@+-]{1,128}$")
_SAFE_EXTRA = re.compile(r"^[A-Za-z0-9._-]{1,64}$")


@dataclass(frozen=True)
class FeedbaxInstallSpec:
    """Structured worker install request for cloud startup scripts."""

    schema_version: str = INSTALL_SPEC_SCHEMA_VERSION
    source: str = "git"
    repository: str = DEFAULT_FEEDBAX_REPOSITORY
    ref: str = DEFAULT_FEEDBAX_REF
    extras: tuple[str, ...] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        if self.schema_version != INSTALL_SPEC_SCHEMA_VERSION:
            raise ValueError(f"unsupported install spec schema_version {self.schema_version!r}")
        if self.source != "git":
            raise ValueError("install spec source must be 'git'")
        if self.repository != DEFAULT_FEEDBAX_REPOSITORY:
            raise ValueError(f"install spec repository must be {DEFAULT_FEEDBAX_REPOSITORY!r}")
        if not _SAFE_REF.fullmatch(self.ref):
            raise ValueError(
                "install spec ref may contain only letters, numbers, '.', '_', '/', "
                "'@', '+', and '-'"
            )
        bad_extras = [extra for extra in self.extras if not _SAFE_EXTRA.fullmatch(extra)]
        if bad_extras:
            raise ValueError(f"invalid install spec extras: {bad_extras!r}")

    @classmethod
    def from_mapping(cls, payload: dict[str, Any] | None) -> "FeedbaxInstallSpec":
        """Build and validate an install spec from a JSON-like payload."""
        if payload is None:
            return cls()
        extras = payload.get("extras", ())
        if extras is None:
            extras = ()
        return cls(
            schema_version=str(payload.get("schema_version", INSTALL_SPEC_SCHEMA_VERSION)),
            source=str(payload.get("source", "git")),
            repository=str(payload.get("repository", DEFAULT_FEEDBAX_REPOSITORY)),
            ref=str(payload.get("ref", DEFAULT_FEEDBAX_REF)),
            extras=tuple(str(extra) for extra in extras),
        )

    def to_pip_target(self) -> str:
        """Return the validated PEP 508 target passed as one pip argument."""
        extras = f"[{','.join(self.extras)}]" if self.extras else ""
        return f"feedbax{extras} @ git+{self.repository}@{self.ref}"


def make_startup_script(
    install_spec: FeedbaxInstallSpec | None = None,
) -> str:
    """Generate the GCP instance startup script.

    Args:
        install_spec: Validated structured install request.

    Returns:
        A bash script string suitable for use as GCP instance metadata.
    """
    spec = install_spec or FeedbaxInstallSpec()
    feedbax_target = shlex.quote(spec.to_pip_target())
    return f"""#!/bin/bash
set -euo pipefail
python -m pip install --upgrade {feedbax_target}
python -m pip install --upgrade uvicorn httpx fastapi 2>&1 | tail -5

cat >/usr/local/bin/feedbax-worker-start <<'WORKER'
#!/bin/bash
set -euo pipefail
metadata_url="http://metadata.google.internal/computeMetadata/v1/instance/attributes"
auth_token="$(curl -fsS -H 'Metadata-Flavor: Google' "$metadata_url/AUTH_TOKEN")"
worker_port="$(curl -fsS -H 'Metadata-Flavor: Google' "$metadata_url/WORKER_PORT")"
if [[ -z "$auth_token" ]]; then
  echo 'Feedbax worker credential metadata is missing' >&2
  exit 1
fi
export FEEDBAX_WORKER_AUTH_TOKEN="$auth_token"
unset auth_token
exec python -m feedbax.web.worker --host 0.0.0.0 --port "$worker_port"
WORKER
chmod 0755 /usr/local/bin/feedbax-worker-start

cat >/etc/systemd/system/feedbax-worker.service <<'UNIT'
[Unit]
Description=Feedbax training worker
After=network-online.target
Wants=network-online.target

[Service]
Type=simple
ExecStart=/usr/local/bin/feedbax-worker-start
Restart=on-failure
RestartSec=5
StandardOutput=append:/var/log/feedbax-worker.log
StandardError=append:/var/log/feedbax-worker.log

[Install]
WantedBy=multi-user.target
UNIT

systemctl daemon-reload
systemctl enable --now feedbax-worker.service
echo "Feedbax worker service started on port ${{WORKER_PORT:-8765}}"
"""
