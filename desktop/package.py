"""Assemble a signed, relocatable macOS Studio release from a clean commit."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import plistlib
import re
import shutil
import subprocess
import sys
import tarfile
import tempfile
import tomllib
from pathlib import Path
from urllib.parse import unquote, urlsplit
from xml.etree import ElementTree

from packaging.markers import Marker
from packaging.tags import compatible_tags, cpython_tags, mac_platforms
from packaging.utils import canonicalize_name, parse_wheel_filename

from seal import (
    APP_NAME,
    FIXTURE_PREFIX,
    LAUNCHER,
    PRODUCTION_BUNDLE_ID,
    RELEASE_SCHEMA,
    json_bytes,
    seal,
    sha256,
)

ROOT = Path(__file__).resolve().parents[1]
DESKTOP = ROOT / "desktop"
SCRATCH = ROOT / ".desktop-build"
TARGET = "aarch64-apple-darwin"
MINIMUM_MACOS = "14.0"
PYTHON_VERSION = "3.12.9"
PYTHON = "Resources/python/bin/python3.12"
PYTHON_ARCHIVE = f"cpython-{PYTHON_VERSION}+20250311-{TARGET}-install_only.tar.gz"
PYTHON_URL = f"https://github.com/astral-sh/python-build-standalone/releases/download/20250311/{PYTHON_ARCHIVE}"
PYTHON_SHA256 = "652e90aeaef68e9c726f883a4c160bbbff205403de68ab2c56fced267c9a652d"
NATIVE_TARGET = SCRATCH / "native"
NATIVE_BINARY = NATIVE_TARGET / TARGET / "release" / "feedbax-studio"
NATIVE_RECEIPT = SCRATCH / "native.json"
TAURI = ("bun", "run", "tauri")
EXTRAS = ("web", "analysis", "training", "mjx")
MJX_WHEELS = frozenset({"jax", "jaxlib", "mujoco-core", "mujoco-mjx"})
LOCKED_BUILD_INPUTS = (
    "desktop/src-tauri", "desktop/shell", "desktop/bun.lock", "desktop/package.json",
    "web/public/icon.svg",
)
WINDOW_SHELL = "vendor/window-shell"
WHEEL_SCHEMA = "feedbax.desktop.wheels.v1"


def run(*arguments: str | Path, cwd: Path = ROOT, env: dict | None = None,
        capture: bool = False, timeout: int = 55) -> str:
    result = subprocess.run(
        [str(argument) for argument in arguments], cwd=cwd, env=env, check=True,
        text=True, stdout=subprocess.PIPE if capture else None, timeout=timeout,
    )
    return result.stdout or ""


def git(*arguments: str) -> str:
    return run("git", *arguments, capture=True).strip()


def clean_revision() -> str:
    if git("status", "--porcelain", "--untracked-files=normal"):
        raise ValueError("desktop releases require a clean, committed input checkout")
    return git("rev-parse", "HEAD^{commit}")


def desktop_icon_source() -> bytes:
    artwork = ElementTree.parse(ROOT / "web/public/icon.svg").getroot()
    artwork.insert(0, ElementTree.Element(
        "{http://www.w3.org/2000/svg}rect", width="100%", height="100%", fill="white",
    ))
    return ElementTree.tostring(artwork)


def checked_window_shell() -> str:
    entry = git("ls-tree", "HEAD", "--", WINDOW_SHELL)
    match = re.fullmatch(rf"160000 commit ([0-9a-f]{{40}})\t{re.escape(WINDOW_SHELL)}", entry)
    if match is None:
        raise ValueError("The shared window shell must be pinned as one Git submodule")
    pin = match[1]
    source = ROOT / WINDOW_SHELL
    if not source.is_dir() or source.resolve() != source:
        raise ValueError("Initialize the pinned shared window shell submodule before building")

    def shell_git(*args: str) -> str:
        return run("git", *args, cwd=source, capture=True).strip()

    if shell_git("rev-parse", "--show-toplevel") != str(source):
        raise ValueError("Initialize the pinned shared window shell submodule before building")
    if shell_git("rev-parse", "HEAD") != pin or shell_git(
        "status", "--porcelain", "--untracked-files=all", "--ignore-submodules=none",
    ):
        raise ValueError("The shared window shell checkout must be clean and match the pinned commit")
    return pin


def native_inputs() -> dict:
    inputs = {path: git("rev-parse", f"HEAD:{path}") for path in LOCKED_BUILD_INPUTS}
    inputs[WINDOW_SHELL] = checked_window_shell()
    return {
        "target": TARGET,
        "minimum_macos": MINIMUM_MACOS,
        "inputs": inputs,
        "desktop_icon_sha256": hashlib.sha256(desktop_icon_source()).hexdigest(),
        "rustc": run("rustc", "-vV", capture=True).strip(),
        "xcode_sdk": run("xcrun", "--sdk", "macosx", "--show-sdk-version", capture=True).strip(),
        "tauri": run(*TAURI, "--version", cwd=DESKTOP, capture=True).strip(),
    }


def native_environment() -> dict:
    return dict(
        os.environ, CARGO_TARGET_DIR=str(NATIVE_TARGET), CARGO_BUILD_JOBS="2",
        CARGO_INCREMENTAL="0", MACOSX_DEPLOYMENT_TARGET=MINIMUM_MACOS,
    )


def build_native(timeout: int) -> None:
    clean_revision()
    inputs = native_inputs()
    icons = DESKTOP / "src-tauri" / "icons"
    stamp = icons / "source.txt"
    icon_source = inputs["desktop_icon_sha256"] + "\n"
    if not all((icons / name).is_file() for name in ("icon.png", "icon.icns", "source.txt")) or stamp.read_text() != icon_source:
        icons.mkdir(parents=True, exist_ok=True)
        source = icons / "source.svg"
        source.write_bytes(desktop_icon_source())
        run(*TAURI, "icon", source, "--output", icons, cwd=DESKTOP)
        stamp.write_text(icon_source)
    run(*TAURI, "build", "--target", TARGET, "--no-bundle", "--ci", "--", "--locked",
        cwd=DESKTOP, env=native_environment(), timeout=timeout)
    if checked_window_shell() != inputs["inputs"][WINDOW_SHELL]:
        raise ValueError("shared window shell pin changed during the native build")
    (SCRATCH / "native.json.tmp").write_bytes(json_bytes(inputs | {"sha256": sha256(NATIVE_BINARY)}))
    (SCRATCH / "native.json.tmp").replace(NATIVE_RECEIPT)


def _locked_wheels(path: Path) -> dict[str, dict]:
    platform_version = tuple(map(int, MINIMUM_MACOS.split(".")))
    platforms = list(mac_platforms(platform_version, "arm64"))
    supported = [
        *cpython_tags((3, 12), platforms=platforms),
        *compatible_tags((3, 12), interpreter="cp312", platforms=platforms),
    ]
    preference = {tag: index for index, tag in enumerate(supported)}
    marker_env = {
        "implementation_name": "cpython", "implementation_version": PYTHON_VERSION,
        "os_name": "posix", "platform_machine": "arm64", "platform_python_implementation": "CPython",
        "platform_release": "23.0.0", "platform_system": "Darwin", "platform_version": "",
        "python_full_version": PYTHON_VERSION, "python_version": "3.12", "sys_platform": "darwin",
    }
    result = {}
    for package in tomllib.loads(path.read_text())["packages"]:
        marker = package.get("marker")
        if marker and not Marker(marker).evaluate(marker_env):
            continue
        candidates = []
        for wheel in package.get("wheels", ()):
            filename = unquote(Path(urlsplit(wheel["url"]).path).name)
            name, version, _, tags = parse_wheel_filename(filename)
            if name != package["name"] or str(version) != package["version"]:
                raise ValueError(f"wheel identity disagrees with lock entry: {filename}")
            ranks = tags.intersection(preference)
            if ranks:
                candidates.append((min(preference[tag] for tag in ranks), filename, wheel))
        if not candidates:
            raise ValueError(f"no macOS arm64 Python {PYTHON_VERSION} wheel for {package['name']}")
        _, filename, wheel = min(candidates, key=lambda item: item[:2])
        if package["name"] in result:
            raise ValueError(f"ambiguous lock entries for {package['name']}")
        result[package["name"]] = {
            "version": package["version"], "filename": filename, "url": wheel["url"],
            "sha256": wheel["hashes"]["sha256"],
        }
    return result


def _numerical_wheels(path: Path, locked: dict[str, dict]) -> dict[str, str]:
    data = path.read_bytes()
    document = json.loads(data)
    if document["schema"] != WHEEL_SCHEMA or document["target"] != TARGET or "runtime" not in document:
        raise ValueError("unsupported numerical wheel manifest")
    provided = set()
    for item in document["wheels"]:
        wheel = (path.parent / item["filename"]).resolve(strict=True)
        if wheel.parent != path.parent or sha256(wheel) != item["sha256"]:
            raise ValueError(f"numerical wheel digest or path mismatch: {wheel.name}")
        name, _, _, _ = parse_wheel_filename(wheel.name)
        if name in provided:
            raise ValueError(f"duplicate numerical wheel for {name}")
        provided.add(name)
        if (
            name not in locked
            or wheel.name != locked[name]["filename"]
            or item["sha256"] != locked[name]["sha256"]
        ):
            raise ValueError(f"numerical wheel differs from the locked macOS CPU profile: {wheel.name}")
        locked[name]["url"] = wheel.as_uri()
    if not MJX_WHEELS <= provided:
        raise ValueError(f"numerical manifest lacks locked MJX CPU wheels: {sorted(MJX_WHEELS - provided)}")
    return {"sha256": hashlib.sha256(data).hexdigest(), "runtime": "mjx-jax-cpu"}


def _release_wheels(stage: Path, manifest: Path) -> tuple[dict[str, dict], dict[str, str]]:
    lock = stage / "pylock.toml"
    run("uv", "export", "--quiet", "--locked", "--no-dev", "--no-emit-project", "--format", "pylock.toml",
        *(argument for extra in EXTRAS for argument in ("--extra", extra)), "--output-file", lock)
    wheels = _locked_wheels(lock)
    if "warp-lang" in wheels:
        raise ValueError("the macOS MJX CPU profile cannot include warp-lang")
    return wheels, _numerical_wheels(manifest, wheels)


def _unpack_python(resources: Path) -> Path:
    archive = SCRATCH / PYTHON_ARCHIVE
    if not archive.exists():
        temporary = archive.with_suffix(".part")
        try:
            run("/usr/bin/curl", "--fail", "--location", "--silent", "--show-error",
                "--output", temporary, PYTHON_URL)
            if sha256(temporary) != PYTHON_SHA256:
                raise ValueError("upstream Python archive SHA-256 mismatch")
            temporary.replace(archive)
        finally:
            temporary.unlink(missing_ok=True)
    if sha256(archive) != PYTHON_SHA256:
        raise ValueError("cached Python archive SHA-256 mismatch")
    with tarfile.open(archive) as package:
        package.extractall(resources, filter="data")
    return resources.parent / PYTHON


def _package_python(
    resources: Path, stage: Path, wheel_dir: Path, wheels: dict[str, dict], numerical: dict[str, str],
) -> dict:
    python = _unpack_python(resources)
    dist = stage / "wheel"
    run("uv", "build", "--wheel", "--python", python, "--require-hashes",
        "--build-constraints", DESKTOP / "build-requirements.txt", "--out-dir", dist)
    (feedbax,) = dist.glob("feedbax-*.whl")
    name, version, _, _ = parse_wheel_filename(feedbax.name)
    wheels[name] = {
        "version": str(version), "filename": feedbax.name, "url": feedbax.as_uri(),
        "sha256": sha256(feedbax),
    }
    requirements = stage / "wheels.txt"
    lines = []
    for name, wheel in sorted(wheels.items()):
        if wheel["url"].startswith("file:"):
            requirement = f"{name}=={wheel['version']}"
        else:
            requirement = f"{name} @ {wheel['url']}"
        lines.append(f"{requirement} --hash=sha256:{wheel['sha256']}\n")
    requirements.write_text("".join(lines))
    site = Path(run(python, "-I", "-B", "-c", "import sysconfig; print(sysconfig.get_path('purelib'))",
                    capture=True).strip())
    if not site.resolve().is_relative_to(resources):
        raise ValueError(f"Python package installation would leave the release: {site}")
    run("uv", "pip", "install", "--python", python, "--target", site, "--no-deps", "--only-binary", ":all:",
        "--no-index", "--find-links", dist, "--find-links", wheel_dir, "--link-mode", "copy",
        "--require-hashes", "-r", requirements)
    run("uv", "pip", "check", "--python", python)
    if (site / "bin").exists():
        shutil.rmtree(site / "bin")
    return {
        "schema": "feedbax.desktop.dependencies.v1",
        "profile": "mjx-cpu",
        "python": {"version": PYTHON_VERSION, "url": PYTHON_URL, "sha256": PYTHON_SHA256},
        "provider": numerical,
        "uv_lock_sha256": sha256(ROOT / "uv.lock"),
        "wheels": {
            canonicalize_name(name): {k: v for k, v in item.items() if k != "url" or v.startswith("https:")}
            for name, item in sorted(wheels.items())
        },
    }


def _write_launch_agent(contents: Path, identifier: str) -> str:
    relative = f"Library/LaunchAgents/{identifier}.runtime.plist"
    path = contents / relative
    path.parent.mkdir(parents=True)
    path.write_bytes(plistlib.dumps({
        "Label": f"{identifier}.runtime",
        "BundleProgram": f"Contents/{LAUNCHER}",
        "ProgramArguments": [Path(LAUNCHER).name, "service", "serve"],
        "RunAtLoad": True,
        "KeepAlive": {"SuccessfulExit": False},
    }))
    return relative


def _rust_notices(resources: Path) -> None:
    metadata = json.loads(run(
        "cargo", "metadata", "--manifest-path", DESKTOP / "src-tauri/Cargo.toml",
        "--locked", "--filter-platform", TARGET, "--format-version", "1",
        "--features", "custom-protocol", capture=True,
    ))
    packages = {package["id"]: package for package in metadata["packages"]}
    nodes = {node["id"]: node for node in metadata["resolve"]["nodes"]}
    root = metadata["resolve"]["root"]
    visited = {root}
    pending = [root]
    inventory = {}
    while pending:
        node = nodes[pending.pop()]
        for dep in node["deps"]:
            identity = dep["pkg"]
            if identity in visited or not any(kind["kind"] is None for kind in dep["dep_kinds"]):
                continue
            visited.add(identity)
            package = packages[identity]
            if any("proc-macro" in target["kind"] for target in package["targets"]):
                continue
            pending.append(identity)
            name = f"{package['name']}@{package['version']}"
            directory = Path(package["manifest_path"]).parent
            files = {
                path for path in directory.iterdir()
                if path.is_file() and re.match(r"^(licen[cs]e|notice|copying)([-._]|$)", path.name, re.I)
            }
            if package["license_file"]:
                files.add(directory / package["license_file"])
            inventory[name] = {
                "license": package["license"],
                "notices": {path.name: path.read_text(errors="replace") for path in sorted(files)},
            }
    (resources / "rust-notices.json").write_bytes(json_bytes(inventory))
    toolchain = Path(run("rustc", "--print", "sysroot", capture=True).strip()) / "share/doc/rust"
    notices = resources / "rust-standard-library-notices"
    shutil.copytree(toolchain / "licenses", notices / "licenses")
    shutil.copyfile(toolchain / "COPYRIGHT-library.html", notices / "COPYRIGHT-library.html")


def package(args: argparse.Namespace) -> None:
    revision = clean_revision()
    manifest_path = args.numerical_manifest.resolve(strict=True)
    version = tomllib.loads((ROOT / "pyproject.toml").read_text())["project"]["version"]
    if not re.fullmatch(r"[1-9][0-9]*(?:\.[0-9]+){0,2}", args.build_number):
        raise ValueError("macOS bundle build number must be a positive numeric version")
    identifier = PRODUCTION_BUNDLE_ID
    fixture = {}
    if args.fixture_root is not None:
        if not args.fixture_name or not re.fullmatch(r"[a-z0-9]+(?:-[a-z0-9]+)*", args.fixture_name):
            raise ValueError("fixture builds require a unique lowercase --fixture-name")
        identifier = FIXTURE_PREFIX + args.fixture_name
        fixture["fixture_root"] = str(args.fixture_root.resolve())
    elif args.fixture_name is not None:
        raise ValueError("--fixture-name requires --fixture-root")
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".studio-package-", dir=output.parent) as temporary:
        stage = Path(temporary)
        wheels, numerical = _release_wheels(stage, manifest_path)
        expected = native_inputs() | {"sha256": sha256(NATIVE_BINARY)}
        if json.loads(NATIVE_RECEIPT.read_bytes()) != expected:
            raise ValueError("native input or binary changed; run the native build phase")
        config = json.dumps({
            "identifier": identifier, "version": version,
            "bundle": {"macOS": {"bundleVersion": args.build_number}},
        })
        run(*TAURI, "bundle", "--target", TARGET, "--bundles", "app", "--no-sign", "--ci",
            "--config", config, cwd=DESKTOP, env=native_environment())
        app = stage / APP_NAME
        (NATIVE_BINARY.parent / "bundle" / "macos" / APP_NAME).rename(app)
        contents = app / "Contents"
        if fixture:
            info = contents / "Info.plist"
            document = plistlib.loads(info.read_bytes())
            document["CFBundleName"] = f"Feedbax Studio Fixture {args.fixture_name}"
            document["CFBundleDisplayName"] = document["CFBundleName"]
            info.write_bytes(plistlib.dumps(document))
        resources = contents / "Resources"
        resources.mkdir(exist_ok=True)
        installer = resources / "installer"
        installer.mkdir()
        for script in ("install.py", "seal.py"):
            shutil.copyfile(DESKTOP / script, installer / script)
        studio = resources / "studio"
        run("bun", "--bun", "vite", "build", "--outDir", studio, cwd=ROOT / "web",
            env=dict(os.environ, VITE_FEEDBAX_DESKTOP="1"))
        dependencies = _package_python(resources, stage, manifest_path.parent, wheels, numerical)
        (resources / "dependencies.json").write_bytes(json_bytes(dependencies))
        _rust_notices(resources)
        launch_agent = _write_launch_agent(contents, identifier)
        shutil.copyfile(ROOT / "LICENSE", resources / "Feedbax-LICENSE")
        manifest = {
            "schema": RELEASE_SCHEMA, "service_protocol": 1, "bundle_id": identifier,
            "version": version, "build_number": args.build_number, "source_commit": revision,
            "platform": {"architecture": "arm64", "minimum_macos": MINIMUM_MACOS},
            "native_build": expected,
            "tooling": {
                "bun": run("bun", "--version", capture=True).strip(),
                "uv": run("uv", "--version", capture=True).strip(),
            },
            "dependency_manifest_sha256": sha256(resources / "dependencies.json"),
            "resources": {
                "launcher": LAUNCHER, "python": PYTHON, "studio": "Resources/studio",
                "dependencies": "Resources/dependencies.json", "launch_agent": launch_agent,
                "installer": "Resources/installer",
            },
        } | fixture
        if clean_revision() != revision:
            raise ValueError("source revision changed during packaging")
        if checked_window_shell() != expected["inputs"][WINDOW_SHELL]:
            raise ValueError("shared window shell pin changed during packaging")
        receipt = seal(app, manifest, args.signing_identity)
        archive = stage / "Feedbax-Studio-macos-arm64.zip"
        run("/usr/bin/ditto", "-c", "-k", "--sequesterRsrc", "--keepParent", app, archive)
        receipt["archive"] = {"name": archive.name, "sha256": sha256(archive), "bytes": archive.stat().st_size}
        result = stage / "release"
        result.mkdir()
        archive.rename(result / archive.name)
        (result / "receipt.json").write_bytes(json_bytes(receipt))
        result.rename(output)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="action", required=True)
    native = sub.add_parser("native")
    native.add_argument("--timeout", type=int, default=55)
    bundle = sub.add_parser("bundle")
    bundle.add_argument("--output", type=Path, required=True)
    bundle.add_argument("--build-number", required=True)
    bundle.add_argument("--signing-identity", required=True)
    bundle.add_argument("--fixture-root", type=Path)
    bundle.add_argument("--fixture-name")
    bundle.add_argument("--numerical-manifest", type=Path, required=True)
    args = parser.parse_args()
    if sys.platform != "darwin" or platform.machine() != "arm64":
        parser.error("packaging requires macOS on Apple Silicon")
    SCRATCH.mkdir(exist_ok=True)
    if args.action == "native":
        build_native(args.timeout)
    else:
        package(args)


if __name__ == "__main__":
    main()
