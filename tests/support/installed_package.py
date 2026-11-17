"""Build and install a local test distribution into an isolated import path."""

from pathlib import Path
import subprocess


def install_test_package(project: Path, root: Path) -> Path:
    wheels = root / "wheels"
    target = root / "site"
    subprocess.run(
        ["uv", "build", "--offline", "--wheel", "--out-dir", str(wheels), str(project)],
        check=True, timeout=45, capture_output=True, text=True,
    )
    (wheel,) = wheels.glob("*.whl")
    subprocess.run(
        ["uv", "pip", "install", "--offline", "--no-deps", "--target", str(target), str(wheel)],
        check=True, timeout=30, capture_output=True, text=True,
    )
    return target
