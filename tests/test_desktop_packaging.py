"""Exercise local packaging inputs without building or signing a release."""

import importlib
from pathlib import Path
import stat
import subprocess
import sys

import pytest


@pytest.fixture
def desktop_package(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "desktop"))
    return importlib.import_module("package")


@pytest.mark.parametrize("documentation_directory", ["rust", "rustc"])
def test_rust_notices_resolve_inside_the_toolchain(desktop_package, tmp_path, documentation_directory):
    docs = tmp_path / "share" / "doc"
    standard_library = docs / documentation_directory
    (standard_library / "licenses").mkdir(parents=True)
    (standard_library / "COPYRIGHT-library.html").touch()
    (docs / "cargo").mkdir()
    (docs / "cargo" / "COPYRIGHT-library.html").touch()
    if documentation_directory == "rustc":
        (docs / "rust").symlink_to("rustc")

    assert desktop_package._rust_standard_library_notice_directory(tmp_path) == standard_library


@pytest.mark.skipif(sys.platform != "darwin", reason="macOS xattr permissions")
def test_bundle_clears_readonly_payload_attributes_without_following_links(
    desktop_package, tmp_path,
):
    seal = importlib.import_module("seal")
    app = tmp_path / seal.APP_NAME
    source = app / "Contents" / "Resources" / "runpod" / "source"
    source.mkdir(parents=True)
    payload = source / "pyproject.toml"
    payload.touch()
    outside = tmp_path / "outside.txt"
    outside.touch()
    (app / "outside-link").symlink_to(outside)
    attribute = "com.feedbax.packaging-test"
    for file in (payload, outside):
        subprocess.run(["/usr/bin/xattr", "-w", attribute, "fixture", file], check=True, timeout=5)
    payload.chmod(0o444)
    source.chmod(0o555)
    try:
        seal.clear_bundle_attributes(app)
        attributes = subprocess.check_output(["/usr/bin/xattr", payload], text=True, timeout=5)
        assert attribute not in attributes.splitlines()
        assert subprocess.check_output(
            ["/usr/bin/xattr", "-p", attribute, outside], text=True, timeout=5,
        ).strip() == "fixture"
        assert stat.S_IMODE(payload.stat().st_mode) == 0o444
        assert stat.S_IMODE(source.stat().st_mode) == 0o555
    finally:
        source.chmod(0o755)
