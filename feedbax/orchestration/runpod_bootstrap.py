"""Standard-library bootstrap sent over a pinned venue connection before import."""

import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import stat
import subprocess
import sys
import tempfile
from collections.abc import Callable


def _private(path: Path) -> None:
    path.mkdir(mode=0o700, parents=True, exist_ok=True)
    info = path.lstat()
    if not stat.S_ISDIR(info.st_mode) or info.st_uid != os.getuid() or info.st_mode & 0o077:
        raise PermissionError(f"Expected a private RunPod session directory: {path}")


def _publish(path: Path, payload: bytes) -> None:
    if path.exists():
        if path.read_bytes() != payload:
            raise ValueError(f"RunPod session already published different bytes at {path.name}")
        return
    with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as stream:
        staged = Path(stream.name)
        try:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
            _link_staged(staged, path)
        finally:
            staged.unlink()


def _link_staged(staged: Path, path: Path) -> None:
    os.link(staged, path)
    fd = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def _uv_path(root: Path) -> Path:
    toolbox = root / "tools"
    _private(toolbox)
    return toolbox / "uv"


def _verified_uv(root: Path, digest: str) -> Path:
    path = _uv_path(root)
    fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
    with os.fdopen(fd, "rb") as stream:
        info = os.fstat(stream.fileno())
        if not stat.S_ISREG(info.st_mode) or info.st_uid != os.getuid() or stat.S_IMODE(info.st_mode) != 0o500:
            raise PermissionError("Expected a private staged uv executable.")
        if hashlib.file_digest(stream, "sha256").hexdigest() != digest:
            raise ValueError("Staged uv differs from the authorized SHA-256.")
    return path


def _stage_uv(root: Path, digest: str) -> None:
    if not re.fullmatch(r"[0-9a-f]{64}", digest):
        raise ValueError("Staging uv requires its exact SHA-256.")
    try:
        _verified_uv(root, digest)
    except FileNotFoundError:
        pass
    else:
        return
    path = _uv_path(root)
    with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as stream:
        staged = Path(stream.name)
        try:
            received = hashlib.sha256()
            while chunk := sys.stdin.buffer.read(1024 * 1024):
                received.update(chunk)
                stream.write(chunk)
            if received.hexdigest() != digest:
                raise ValueError("Transferred uv differs from the authorized SHA-256.")
            stream.flush()
            os.fchmod(stream.fileno(), 0o500)
            os.fsync(stream.fileno())
            _link_staged(staged, path)
        finally:
            staged.unlink()


def _prepare(root: Path, settings: dict) -> None:
    _private(root / "incoming")
    if "watchdog" not in settings:
        return
    watchdog = root / "watchdog.py"
    config = root / "watchdog.json"
    _publish(watchdog, settings["watchdog"].encode())
    _publish(config, json.dumps(settings["policy"], sort_keys=True).encode())
    subprocess.run(
        [sys.executable, "-I", str(watchdog), "--config", str(config), "--detach"],
        stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, check=True,
        env=os.environ | settings["worker_env"], timeout=5,
    )


_PROBE = """
import json
import jax
import sys
from pathlib import Path
from feedbax.orchestration.revision import assert_feedbax_source_residence

assert_feedbax_source_residence(Path(sys.argv[1]))
backend = jax.default_backend()
if backend != sys.argv[2]:
    raise ValueError(f"Expected JAX backend {sys.argv[2]!r}, observed {backend!r}")
devices = [device.device_kind for device in jax.devices()]
proof = int(jax.jit(lambda x: x * x + 1)(2).block_until_ready())
if not devices or proof != 5:
    raise ValueError("The JAX device probe did not complete")
print(json.dumps({"jax": jax.__version__, "backend": backend, "devices": devices,
                  "python": sys.executable, "probe_result": proof}))
"""


def _install(root: Path, settings: dict, verify_tree: Callable[[Path], tuple[str, int]]) -> None:
    cached_probe = root / "probe.json"
    if cached_probe.exists():
        sys.stdout.buffer.write(cached_probe.read_bytes())
        return

    incoming, repo = root / "incoming", root / "repo"
    expected = settings["content_sha256"], settings["file_count"]
    if not repo.exists():
        if verify_tree(incoming) != expected:
            raise ValueError("Transferred repository differs from the approved snapshot")
        os.rename(incoming, repo)
    env = {key: value for key, value in os.environ.items() if key not in ("VIRTUAL_ENV", "PYTHONPATH")}
    env.update(settings["worker_env"])
    env.update(
        UV_PROJECT_ENVIRONMENT=str(root / "venv"), UV_PYTHON_DOWNLOADS="never", PYTHONDONTWRITEBYTECODE="1",
    )
    digest = settings.get("uv_sha256")
    uv = str(_verified_uv(root, digest)) if digest is not None else settings["uv"]
    version = subprocess.check_output([uv, "--version"], env=env, timeout=5).decode().split()
    if version[:2] != ["uv", settings["uv_version"]]:
        raise ValueError(f"Expected uv {settings['uv_version']!r}")
    command = [uv, "sync", "--locked", "--no-default-groups", "--python", sys.executable]
    for group in settings["groups"]:
        command.extend(("--group", group))
    for extra in settings["extras"]:
        command.extend(("--extra", extra))
    subprocess.run(command, cwd=repo, env=env, stdout=sys.stderr, check=True, timeout=settings["timeout"])
    python = root / "venv" / "bin" / "python"
    probe = subprocess.check_output(
        [str(python), "-I", "-B", "-c", _PROBE, str(repo), settings["backend"]],
        cwd=repo, env=env, timeout=30,
    )
    _publish(cached_probe, probe)
    sys.stdout.buffer.write(probe)


def _complete(root: Path, settings: dict) -> None:
    _publish(root / "completion-requested", b"")
    realized = root / "realization.json"
    if not realized.exists():
        python = root / "venv" / "bin" / "python"
        subprocess.run(
            [str(python), "-I", "-B", "-m", "feedbax.host", "--root", str(root / "host"), "init-session"],
            input=json.dumps(settings["fence"]).encode(), stdout=sys.stderr, check=True, timeout=10,
        )
    _publish(realized, settings["realization"].encode())
    _publish(root / "ready", b"")


def bootstrap(verify_tree: Callable[[Path], tuple[str, int]]) -> None:
    action, location, *arguments = sys.argv[1:]
    if len(arguments) != (1 if action == "stage-uv" else 0):
        raise ValueError("Unexpected RunPod bootstrap arguments.")
    root = Path(location)
    if action == "read":
        try:
            sys.stdout.buffer.write((root / "realization.json").read_bytes())
        except FileNotFoundError:
            pass
        return
    _private(root)
    settings = json.load(sys.stdin.buffer) if action != "stage-uv" else {}
    with (root / "setup.lock").open("ab") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if action == "close-opening":
            unopened = not any(
                (root / name).exists() for name in ("completion-requested", "realization.json", "host")
            )
            if unopened:
                _publish(root / "opening-aborted", b"")
            print(json.dumps(unopened))
            return
        if (root / "opening-aborted").exists():
            raise ValueError("RunPod session setup was closed before publication.")
        if action == "stage-uv":
            _stage_uv(root, arguments[0])
            return
        if action == "prepare":
            _prepare(root, settings)
            digest = settings.get("uv_sha256")
            send_uv = False
            if digest is not None:
                try:
                    _verified_uv(root, digest)
                except FileNotFoundError:
                    send_uv = True
            print(json.dumps({"send": not (root / "repo").exists(), "send_uv": send_uv}))
        elif action == "install":
            _install(root, settings, verify_tree)
        elif action == "complete":
            _complete(root, settings)
        else:
            raise ValueError(f"Unknown RunPod bootstrap operation: {action!r}")
