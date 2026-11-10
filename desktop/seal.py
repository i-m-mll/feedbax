"""Sign and verify the exact macOS Studio app payload."""

from __future__ import annotations

import hashlib
import json
import os
import plistlib
import re
import stat
import subprocess
from collections.abc import Iterator
from pathlib import Path
from typing import Any

APP_NAME = "Feedbax Studio.app"
RELEASE_SCHEMA = "feedbax.desktop.release.v1"
RECEIPT_SCHEMA = "feedbax.desktop.receipt.v1"
PRODUCTION_BUNDLE_ID = "org.feedbax.studio"
FIXTURE_PREFIX = f"{PRODUCTION_BUNDLE_ID}.fixture."
LAUNCHER = "MacOS/feedbax-studio"
MANIFEST = "Resources/release.json"
_MACH_O_MAGIC = {
    bytes.fromhex(value) for value in ("cffaedfe", "feedfacf", "cafebabe", "cafebabf")
}
_BUILD_RPATHS = {
    "scipy": frozenset({
        "/opt/homebrew/Cellar/gcc@13/13.4.0/lib/gcc/13",
        "/opt/homebrew/Cellar/gcc@13/13.4.0/lib/gcc/13/gcc",
        "/opt/homebrew/Cellar/gcc@13/13.4.0/lib/gcc/13/gcc/aarch64-apple-darwin23/13",
    }),
    "sklearn": frozenset({"/Users/runner/miniconda3/envs/build/lib"}),
    "mujoco": frozenset({"/Volumes/BuildData/foundry_bot/w/src/release"}),
}


def sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def json_bytes(document: object) -> bytes:
    return (json.dumps(document, indent=2, sort_keys=True) + "\n").encode()


def command(*args: str | Path, timeout: int = 45) -> str:
    result = subprocess.run(
        [str(arg) for arg in args],
        check=True,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        timeout=timeout,
    )
    return result.stdout


def _entries(root: Path) -> Iterator[tuple[str, Path]]:
    for path in sorted(root.rglob("*")):
        yield path.relative_to(root).as_posix(), path


def _inventory(contents: Path, *, sealed: bool) -> dict[str, dict[str, Any]]:
    records = {}
    for relative, path in _entries(contents):
        if not sealed and (
            relative in (MANIFEST, LAUNCHER)
            or relative == "_CodeSignature"
            or relative.startswith("_CodeSignature/")
        ):
            continue
        info = path.lstat()
        mode = stat.S_IMODE(info.st_mode)
        if path.is_symlink():
            target = os.readlink(path)
            if Path(target).is_absolute() or not path.resolve(strict=True).is_relative_to(contents):
                raise ValueError(f"payload symlink leaves the installed release: {relative}")
            record = {"link": target}
        elif path.is_file():
            record = {"sha256": sha256(path), "size": info.st_size, "mode": mode}
        elif path.is_dir():
            continue
        else:
            raise ValueError(f"unsupported payload entry {relative}")
        records[relative] = record
    return records


def _macho_files(contents: Path) -> list[Path]:
    result = []
    for _, path in _entries(contents):
        if path.is_symlink() or not path.is_file():
            continue
        with path.open("rb") as stream:
            if stream.read(4) in _MACH_O_MAGIC:
                result.append(path)
    return result


def _loader_references(block: str) -> set[tuple[str, str]]:
    references = set()
    for load_command in re.split(r"^Load command \d+\n", block, flags=re.M):
        operation = re.search(r"^\s*cmd (LC_\w+)$", load_command, re.M)
        reference = re.search(r"^\s*(?:name|path) (.+?) \(offset", load_command, re.M)
        if operation is None or reference is None:
            continue
        kind = operation[1]
        if kind == "LC_RPATH" or (kind.endswith("_DYLIB") and kind != "LC_ID_DYLIB"):
            references.add((kind, reference[1]))
    return references


def relocate_and_inventory_native(contents: Path, minimum: str) -> dict[str, dict[str, Any]]:
    """Record arm64 deployment targets and reject external developer dylib paths."""
    files = _macho_files(contents)
    records = {}
    for start in range(0, len(files), 50):
        batch = files[start:start + 50]
        output = command("xcrun", "otool", "-arch", "arm64", "-l", *batch)
        for index, path in enumerate(batch):
            marker = f"{path}:\n"
            if marker not in output:
                raise ValueError(f"missing arm64 load commands for {path}")
            block = output.split(marker, 1)[1]
            if index + 1 < len(batch):
                block = block.split(f"{batch[index + 1]}:\n", 1)[0]
            versions = re.findall(r"\bminos ([0-9.]+)", block)
            if not versions and "LC_VERSION_MIN_MACOSX" in block:
                versions = re.findall(
                    r"LC_VERSION_MIN_MACOSX\s+cmdsize \d+\s+version ([0-9.]+)", block
                )
            if len(versions) != 1:
                raise ValueError(f"cannot prove a single macOS deployment target for {path}")
            def version_parts(value: str) -> tuple[int, ...]:
                return tuple(int(part) for part in (value + ".0.0").split(".")[:3])
            if version_parts(versions[0]) > version_parts(minimum):
                raise ValueError(f"{path} requires macOS {versions[0]}, above declared {minimum}")
            relative = path.relative_to(contents)
            parts = relative.parts
            in_site_packages = (
                len(parts) > 5 and parts[:3] == ("Resources", "python", "lib")
                and parts[4] == "site-packages"
            )
            package = parts[5] if in_site_packages else ""
            references = _loader_references(block)
            removed = sorted(
                reference for kind, reference in references
                if kind == "LC_RPATH" and reference in _BUILD_RPATHS.get(package, ())
            )
            if removed:
                command(
                    "xcrun", "install_name_tool",
                    *(argument for reference in removed for argument in ("-delete_rpath", reference)),
                    path,
                )
                actual = _loader_references(command("xcrun", "otool", "-arch", "arm64", "-l", path))
                if actual != references - {("LC_RPATH", reference) for reference in removed}:
                    raise ValueError(f"unexpected loader changes while relocating {relative}")
                references = actual
            for _, reference in references:
                if not reference.startswith((
                    "/System/Library/", "/usr/lib/", "@loader_path/", "@executable_path/", "@rpath/"
                )) and reference not in ("@loader_path", "@executable_path"):
                    raise ValueError(f"{path} loads a library outside the app or OS: {reference}")
            records[relative.as_posix()] = {
                "architecture": "arm64",
                "minimum_macos": versions[0],
                "load_references": sorted({reference for _, reference in references}),
                **({"removed_build_rpaths": removed} if removed else {}),
            }
    if LAUNCHER not in records:
        raise ValueError("the Tauri launcher is not a native arm64 executable")
    return records


def resolve_identity(value: str) -> tuple[str, str]:
    if value == "-":
        return value, "ad-hoc"
    matches = []
    for fingerprint, name in re.findall(
        r'^\s*\d+\) ([0-9A-F]{40}) "([^"\n]+)"',
        command("security", "find-identity", "-v", "-p", "codesigning"),
        re.M,
    ):
        if value in (fingerprint, name):
            matches.append((fingerprint, name))
    if len(matches) != 1:
        raise ValueError("configured signing identity must select exactly one valid certificate")
    fingerprint, name = matches[0]
    if name.startswith("Apple Development: "):
        return fingerprint, "apple-development"
    raise ValueError("local packaging supports an Apple Development identity or explicit ad-hoc")


def seal(app: Path, manifest: dict[str, Any], signing_identity: str) -> dict[str, Any]:
    contents = app / "Contents"
    identity, classification = resolve_identity(signing_identity)
    if (contents / MANIFEST).exists() or (contents / "_CodeSignature").exists():
        raise ValueError("sealing requires a new, unsigned Tauri bundle")
    command("xattr", "-cr", app)
    native = relocate_and_inventory_native(contents, manifest["platform"]["minimum_macos"])
    for relative in native:
        if relative != LAUNCHER:
            command("codesign", "--force", "--sign", identity, "--timestamp=none", contents / relative)
    manifest["native"] = native
    manifest["inventory"] = _inventory(contents, sealed=False)
    manifest["signing"] = {
        "classification": classification,
        "identity": identity,
        "hardened_runtime": False,
        "notarization": "not-requested",
    }
    (contents / MANIFEST).write_bytes(json_bytes(manifest))
    command("codesign", "--force", "--sign", identity, "--timestamp=none", app)
    return verify(app)


def verify(app: Path) -> dict[str, Any]:
    """Validate the outer Apple seal and the complete manifest inventory."""
    contents = app.resolve(strict=True) / "Contents"
    manifest_path = contents / MANIFEST
    manifest = json.loads(manifest_path.read_bytes())
    if manifest.get("schema") != RELEASE_SCHEMA or manifest.get("service_protocol") != 1:
        raise ValueError("unsupported desktop release manifest")
    info = plistlib.loads((contents / "Info.plist").read_bytes())
    bundle_id = manifest["bundle_id"]
    if (
        info.get("CFBundleIdentifier") != bundle_id
        or info.get("CFBundleExecutable") != Path(LAUNCHER).name
        or info.get("CFBundleVersion") != manifest["build_number"]
    ):
        raise ValueError("Info.plist does not identify the declared release")
    if bundle_id == PRODUCTION_BUNDLE_ID:
        if "fixture_root" in manifest:
            raise ValueError("production release cannot declare a fixture root")
    elif not bundle_id.startswith(FIXTURE_PREFIX) or not Path(manifest["fixture_root"]).is_absolute():
        raise ValueError("fixture release requires an isolated absolute instance root")
    for role, relative in manifest["resources"].items():
        resource = contents / relative
        if Path(relative).is_absolute() or ".." in Path(relative).parts:
            raise ValueError(f"invalid installed resource role {role}")
        if not resource.resolve(strict=True).is_relative_to(contents):
            raise ValueError(f"installed resource leaves the release: {role}")
    if sha256(contents / manifest["resources"]["dependencies"]) != manifest["dependency_manifest_sha256"]:
        raise ValueError("installed dependency manifest does not match its declared identity")
    command("codesign", "--verify", "--deep", "--strict", app)
    # Code-signing verification alone does not account for mode or extra files.
    inventory = _inventory(contents, sealed=True)
    payload = {path: value for path, value in inventory.items()
               if path not in (MANIFEST, LAUNCHER) and not path.startswith("_CodeSignature/")}
    if payload != manifest["inventory"]:
        raise ValueError("sealed payload does not match the release inventory")
    description = command("codesign", "--display", "--verbose=4", app)
    cdhash = re.search(r"^CDHash=([a-f0-9]+)$", description, re.M)
    team = re.search(r"^TeamIdentifier=(.+)$", description, re.M)
    authorities = re.findall(r"^Authority=(.+)$", description, re.M)
    classification = (
        "apple-development" if authorities and authorities[0].startswith("Apple Development: ")
        else "ad-hoc" if "Signature=adhoc" in description else "unrecognized"
    )
    if cdhash is None or classification != manifest["signing"]["classification"]:
        raise ValueError("outer signature class or CDHash does not match the release policy")
    return {
        "schema": RECEIPT_SCHEMA,
        "bundle_id": bundle_id,
        "version": manifest["version"],
        "build_number": manifest["build_number"],
        "source_commit": manifest["source_commit"],
        "manifest_sha256": inventory[MANIFEST]["sha256"],
        "dependency_manifest_sha256": manifest["dependency_manifest_sha256"],
        "sealed_tree_sha256": hashlib.sha256(json_bytes(inventory)).hexdigest(),
        "sealed_file_bytes": sum(item.get("size", 0) for item in inventory.values()),
        "signing": {
            "classification": classification,
            "cdhash": cdhash[1],
            "team": team[1] if team else None,
            "notarization": "not-requested",
        },
    }
