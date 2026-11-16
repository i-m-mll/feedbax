"""Stage the source and Linux toolchain shipped with a Studio release."""

from __future__ import annotations

import argparse
import hashlib
import shutil
from pathlib import Path
from tempfile import TemporaryDirectory

from feedbax.orchestration.repo_snapshot import seal_repo_snapshot
from feedbax.orchestration.runpod_session import RunPodEnvironment

UV_VERSION = "0.13.8"
UV_SHA256 = "4a7607589912d9c20e86b860b032f7c7f2913c9a3e76f46aef136885387452b8"


def stage_runpod_venue(source: Path, destination: Path, revision: str, uv_linux: Path) -> None:
    destination.mkdir()
    toolchain = destination / "uv"
    shutil.copyfile(uv_linux, toolchain)
    with toolchain.open("rb") as stream:
        if hashlib.file_digest(stream, "sha256").hexdigest() != UV_SHA256:
            raise ValueError(f"RunPod requires the qualified Linux x86-64 uv {UV_VERSION} binary.")
    with TemporaryDirectory(prefix=".runpod-source-", dir=destination.parent) as directory:
        snapshot = seal_repo_snapshot(
            "feedbax", source, snapshot_parent=directory, exclude_gitlinks=("vendor/window-shell",),
        )
        if snapshot.record.dirty or snapshot.record.commit != revision:
            raise ValueError("The RunPod venue source differs from the Studio release commit.")
        snapshot.staging_root.chmod(0o755)
        snapshot.staging_root.rename(destination / "source").chmod(0o555)
    environment = RunPodEnvironment(
        workspace="/root/feedbax-m4", bootstrap_python="/usr/bin/python3.12",
        uv_version=UV_VERSION, uv_sha256=UV_SHA256, snapshot=snapshot.record,
        backend="gpu", groups=("mjx-warp",), extras=("web",),
    )
    (destination / "environment.json").write_text(environment.model_dump_json() + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("destination", type=Path)
    parser.add_argument("revision")
    parser.add_argument("uv_linux", type=Path)
    arguments = parser.parse_args()
    stage_runpod_venue(arguments.source, arguments.destination, arguments.revision, arguments.uv_linux)


if __name__ == "__main__":
    main()
