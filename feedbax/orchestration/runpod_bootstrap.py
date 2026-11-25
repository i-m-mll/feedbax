"""Standard-library bootstrap sent over a pinned venue connection before import."""

import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
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
from feedbax.revision import assert_feedbax_source_residence

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


_PACKAGE_VERSIONS = """
from importlib.metadata import distributions

for distribution in distributions():
    print(f"{distribution.name}=={distribution.version}")
"""


_PACKAGE_VERIFY = """
from importlib.metadata import distribution
import json
from pathlib import Path
import sys
from urllib.parse import unquote, urlsplit

for pin, source in json.loads(sys.argv[1]):
    installed = distribution(pin["distribution"])
    direct = urlsplit(json.loads(installed.read_text("direct_url.json") or "{}").get("url", ""))
    if (
        installed.version != pin["version"] or direct.scheme != "file" or direct.netloc
        or Path(unquote(direct.path)).resolve() != Path(source).resolve()
        or not any(entry.group == "feedbax.plugins" for entry in installed.entry_points)
    ):
        raise ValueError(f"Installed project package differs from its approved source: {pin['distribution']}")
"""


def _package_copies(root: Path, packages: list[dict]) -> list[str]:
    if not packages:
        return []
    directory = root / "project-packages"
    _private(directory)
    missing = []
    for digest in sorted({pin["snapshot"]["content_sha256"] for pin in packages}):
        if not re.fullmatch(r"[a-f0-9]{64}", digest):
            raise ValueError("Project package has an invalid snapshot digest.")
        if not (directory / digest).exists():
            _private(directory / (digest + ".incoming"))
            missing.append(digest)
    return missing


def _install_project_packages(root, settings, verify_tree, uv, python, env) -> None:
    packages = settings.get("project_packages", [])
    if not packages:
        return
    with tempfile.TemporaryDirectory(prefix="build-packages-", dir=root) as temporary:
        build_root = Path(temporary)
        sources = []
        for pin in packages:
            snapshot = pin["snapshot"]
            digest = snapshot["content_sha256"]
            sealed = root / "project-packages" / digest
            if not sealed.exists():
                incoming = sealed.with_suffix(".incoming")
                if verify_tree(incoming) != (digest, snapshot["file_count"]):
                    raise ValueError(f"Transferred project package differs from its approved snapshot: {pin['distribution']}")
                os.rename(incoming, sealed)
            elif verify_tree(sealed) != (digest, snapshot["file_count"]):
                raise ValueError(f"Staged project package differs from its approved snapshot: {pin['distribution']}")
            build = build_root / digest
            if not build.exists():
                shutil.copytree(sealed, build, symlinks=True)
                for directory, _, _ in os.walk(build):
                    Path(directory).chmod(0o700)
            sources.append((pin, str(build / pin["project_dir"])))
        constraints = build_root / "constraints.txt"
        constraints.write_bytes(subprocess.check_output(
            [str(python), "-I", "-B", "-c", _PACKAGE_VERSIONS], env=env, timeout=10,
        ))
        command = [uv, "pip", "install", "--python", str(python), "--constraint", str(constraints)]
        for pin, _ in sources:
            command.extend(("--reinstall-package", pin["distribution"]))
        command.extend(source for _, source in sources)
        subprocess.run(command, cwd=build_root, env=env, stdout=sys.stderr, check=True, timeout=settings["timeout"])
        subprocess.run(
            [str(python), "-I", "-B", "-c", _PACKAGE_VERIFY, json.dumps(sources)],
            env=env, check=True, timeout=10,
        )

    environment = root / "environment"
    _private(environment)
    encoded_pins = json.dumps(packages, sort_keys=True)
    lines = ["version = 1", f"project-package-pins = {json.dumps(encoded_pins)}"]
    installed = subprocess.check_output(
        [str(python), "-I", "-B", "-c", _PACKAGE_VERSIONS], env=env, text=True, timeout=10,
    )
    for requirement in sorted(installed.splitlines()):
        name, version = requirement.split("==", 1)
        lines.extend(("", "[[package]]", f"name = {json.dumps(name)}", f"version = {json.dumps(version)}"))
    _publish(environment / "uv.lock", ("\n".join(lines) + "\n").encode())


def _install(root: Path, settings: dict, verify_tree: Callable[[Path], tuple[str, int]]) -> None:
    cached_probe = root / "probe.json"
    packages = settings.get("project_packages", [])
    if cached_probe.exists():
        probe = cached_probe.read_bytes()
        if json.loads(probe).get("project_packages", []) != packages:
            raise ValueError("The completed install identifies different project packages.")
        sys.stdout.buffer.write(probe)
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
    _install_project_packages(root, settings, verify_tree, uv, python, env)
    probe = subprocess.check_output(
        [str(python), "-I", "-B", "-c", _PROBE, str(repo), settings["backend"]],
        cwd=repo, env=env, timeout=30,
    )
    if packages:
        probe = json.dumps(json.loads(probe) | {"project_packages": packages}).encode()
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
            send_packages = _package_copies(root, settings.get("project_packages", []))
            print(json.dumps({"send": not (root / "repo").exists(), "send_uv": send_uv, "send_packages": send_packages}))
        elif action == "install":
            _install(root, settings, verify_tree)
        elif action == "complete":
            _complete(root, settings)
        else:
            raise ValueError(f"Unknown RunPod bootstrap operation: {action!r}")
