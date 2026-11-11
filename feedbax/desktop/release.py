"""Read the immutable release manifest and resolve per-user storage for Studio."""

from __future__ import annotations

import hashlib
import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

RELEASE_SCHEMA = "feedbax.desktop.release.v1"
SERVICE_PROTOCOL = 1
PRODUCTION_BUNDLE_ID = "org.feedbax.studio"
FIXTURE_BUNDLE_PREFIX = f"{PRODUCTION_BUNDLE_ID}.fixture."


@dataclass(frozen=True)
class Release:
    contents: Path
    document: dict[str, Any]
    digest: str

    @classmethod
    def load(cls, contents: Path) -> Release:
        contents = contents.resolve(strict=True)
        data = (contents / "Resources/release.json").read_bytes()
        document = json.loads(data)
        if document.get("schema") != RELEASE_SCHEMA:
            raise ValueError("unsupported desktop release manifest")
        if document.get("service_protocol") != SERVICE_PROTOCOL:
            raise ValueError("unsupported desktop service protocol")
        bundle = document.get("bundle_id")
        if not isinstance(bundle, str) or not (
            bundle == PRODUCTION_BUNDLE_ID or bundle.startswith(FIXTURE_BUNDLE_PREFIX)
        ):
            raise ValueError("unsupported desktop bundle identity")
        return cls(contents, document, hashlib.sha256(data).hexdigest())

    def resource(self, role: str) -> Path:
        relative = Path(self.document["resources"][role])
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError(f"release resource {role!r} must be relative to Contents")
        path = (self.contents / relative).resolve(strict=True)
        if not path.is_relative_to(self.contents):
            raise ValueError(f"release resource {role!r} leaves the installed program")
        return path

    @property
    def fixture_root(self) -> Path | None:
        raw = self.document.get("fixture_root")
        is_fixture = self.document["bundle_id"].startswith(FIXTURE_BUNDLE_PREFIX)
        if is_fixture and isinstance(raw, str) and Path(raw).is_absolute():
            return Path(raw)
        if is_fixture or raw is not None:
            raise ValueError("desktop fixtures require an explicit isolated absolute root")
        return None


@dataclass(frozen=True)
class AppPaths:
    state: Path
    config: Path
    settings: Path
    cache: Path
    web: Path
    workspace: Path

    @classmethod
    def resolve(cls, *, instance_root: Path | None = None) -> AppPaths:
        home = Path.home()
        if instance_root is not None:
            if not instance_root.is_absolute():
                raise ValueError("desktop instance root must be absolute")
            state, cache = instance_root / "state", instance_root / "cache"
        elif sys.platform == "win32":
            state = _windows_local_app_data() / "Feedbax"
            cache = state / "Cache"
        else:
            state = home / ".feedbax"
            xdg_cache = Path(os.environ.get("XDG_CACHE_HOME", ""))
            cache_home = (
                xdg_cache if sys.platform == "linux" and xdg_cache.is_absolute() else home / ".cache"
            )
            cache = cache_home / "feedbax"
        return cls(
            state=state,
            config=state / "config",
            settings=state / "settings",
            cache=cache,
            web=state / "web",
            workspace=state / "workspace",
        )

    @property
    def control(self) -> Path:
        return self.config / "desktop-service"

    @property
    def logs(self) -> Path:
        return self.cache / "logs"

    def configure_studio(self, *, desktop: bool = True) -> None:
        """Bind the existing Studio authorities before loading its Python modules."""
        os.environ.update(
            FEEDBAX_WEB_DATA=str(self.web),
            FEEDBAX_TRAJECTORIES_DIR=str(self.state / "trajectories"),
            FEEDBAX_EXPERIMENTS_CONFIG_DIR=str(self.config),
            FEEDBAX_ORCHESTRATION_ROOT=str(self.workspace / "orchestration"),
            FEEDBAX_CONTROLLER_EVENT_LOG=str(self.state / "controller" / "events.jsonl"),
            FEEDBAX_RUNS_DIR=str(self.workspace / "runs"),
            FEEDBAX_JAX_COMPILATION_CACHE_DIR=str(self.cache / "jax"),
            JAX_COMPILATION_CACHE_DIR=str(self.cache / "jax"),
            JAX_PLATFORMS="cpu",
            TMPDIR=str(self.cache / "tmp"),
            PYTHONDONTWRITEBYTECODE="1",
        )
        if desktop:
            os.environ["FEEDBAX_DESKTOP_STATE"] = str(self.state)
        else:
            os.environ.pop("FEEDBAX_DESKTOP_STATE", None)
        for directory in (self.control, self.settings, self.workspace, self.cache / "tmp"):
            directory.mkdir(mode=0o700, parents=True, exist_ok=True)
        control_stat = self.control.stat(follow_symlinks=False)
        if not self.control.is_dir() or self.control.is_symlink():
            raise ValueError(f"desktop control path is not a private directory: {self.control}")
        if hasattr(os, "getuid") and (
            control_stat.st_uid != os.getuid() or control_stat.st_mode & 0o077
        ):
            raise PermissionError(f"desktop control directory must be owner-only: {self.control}")


def _windows_local_app_data() -> Path:
    import ctypes
    import uuid
    from ctypes import wintypes

    # FOLDERID_LocalAppData from KnownFolders.h.
    folder_id = (ctypes.c_ubyte * 16).from_buffer_copy(
        uuid.UUID("F1B32785-6FBA-4FCF-9D55-7B8E7F157091").bytes_le
    )
    result = ctypes.c_wchar_p()
    lookup = ctypes.windll.shell32.SHGetKnownFolderPath
    lookup.argtypes = [
        ctypes.POINTER(ctypes.c_ubyte),
        wintypes.DWORD,
        wintypes.HANDLE,
        ctypes.POINTER(ctypes.c_wchar_p),
    ]
    if lookup(folder_id, 0, None, ctypes.byref(result)) != 0:
        raise OSError("could not resolve the LocalAppData Known Folder")
    try:
        return Path(result.value)
    finally:
        ctypes.windll.ole32.CoTaskMemFree(result)
