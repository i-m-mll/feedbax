"""Standard-library bootstrap sent over a pinned venue connection before import."""

import fcntl
import json
import os
from pathlib import Path
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
            os.link(staged, path)
        finally:
            staged.unlink()
    fd = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


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
    uv = settings["uv"]
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
        [str(python), "-I", "-c", _PROBE, str(repo), settings["backend"]],
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
            [str(python), "-I", "-m", "feedbax.host", "--root", str(root / "host"), "init-session"],
            input=json.dumps(settings["fence"]).encode(), stdout=sys.stderr, check=True, timeout=10,
        )
    _publish(realized, settings["realization"].encode())
    _publish(root / "ready", b"")


def bootstrap(verify_tree: Callable[[Path], tuple[str, int]]) -> None:
    action, location = sys.argv[1:]
    root = Path(location)
    if action == "read":
        try:
            sys.stdout.buffer.write((root / "realization.json").read_bytes())
        except FileNotFoundError:
            pass
        return
    _private(root)
    settings = json.load(sys.stdin.buffer)
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
        if action == "prepare":
            _prepare(root, settings)
            print(json.dumps({"send": not (root / "repo").exists()}))
        elif action == "install":
            _install(root, settings, verify_tree)
        elif action == "complete":
            _complete(root, settings)
        else:
            raise ValueError(f"Unknown RunPod bootstrap operation: {action!r}")
