"""Exercise local packaging inputs without building or signing a release."""

import importlib
from importlib.util import source_from_cache
import json
import os
from pathlib import Path
import plistlib
import signal
import stat
import subprocess
import sys
from tempfile import TemporaryDirectory

import pytest


@pytest.fixture
def desktop_package(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "desktop"))
    return importlib.import_module("package")


@pytest.mark.parametrize("fixture", [False, True])
def test_background_app_owns_no_documents_and_keeps_the_registered_agent(desktop_package, tmp_path, monkeypatch, fixture):
    seal = importlib.import_module("seal")
    contents = tmp_path / seal.APP_NAME / "Contents"
    resources = contents / "Resources"
    resources.mkdir(parents=True)
    icon = "icon.icns"
    (resources / icon).write_bytes(b"icon")
    (contents / "Info.plist").write_bytes(plistlib.dumps({"CFBundleIconFile": icon}))
    binary = tmp_path / "built-helper"
    binary.write_bytes(b"executable")
    monkeypatch.setattr(desktop_package, "BACKGROUND_BINARY", binary)
    bundle = f"{seal.FIXTURE_PREFIX}example" if fixture else seal.PRODUCTION_BUNDLE_ID

    agent = desktop_package._write_background_activity(contents, bundle, "0.2.0", "4")
    helper = contents / seal.BACKGROUND_APP
    manifest = {
        "bundle_id": bundle, "build_number": "4", "version": "0.2.0",
        "platform": {"minimum_macos": desktop_package.MINIMUM_MACOS},
        "resources": {"launch_agent": agent, "background_activity": seal.BACKGROUND_PROGRAM},
    }
    assert seal._verify_background_layout(contents, manifest) == helper
    agent_plist = plistlib.loads((contents / agent).read_bytes())
    assert agent_plist["Label"] == f"{bundle}.runtime"
    assert agent_plist["RunAtLoad"] and agent_plist["KeepAlive"] == {"SuccessfulExit": False}
    assert (tmp_path / seal.APP_NAME / agent_plist["BundleProgram"]).read_bytes() == binary.read_bytes()

    info_path = helper / "Contents" / "Info.plist"
    info = plistlib.loads(info_path.read_bytes())
    assert info["CFBundleIdentifier"] == f"{bundle}.background-activity"
    info["CFBundleDocumentTypes"] = [{"CFBundleTypeName": "Project"}]
    info_path.write_bytes(plistlib.dumps(info))
    with pytest.raises(ValueError, match="helper identity"):
        seal._verify_background_layout(contents, manifest)


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


@pytest.mark.skipif(sys.platform != "darwin", reason="macOS bundled Python")
def test_bundled_cold_cli_preserves_bundle_contents(desktop_package, tmp_path):
    bundle = os.environ.get("FEEDBAX_TEST_STUDIO_APP")
    if not bundle:
        pytest.skip("Open a private Studio archive and set FEEDBAX_TEST_STUDIO_APP to its app path")
    from feedbax.contracts.project_io import ProjectConfiguration, dump_text_project
    from feedbax.host import HostClient
    from tests.support.managed_train_fixture import managed_train_project
    from tests.test_local_job_host import wait_until

    seal = importlib.import_module("seal")
    with TemporaryDirectory(dir=tmp_path) as scratch:
        root = Path(scratch).resolve()
        app = root / desktop_package.APP_NAME
        subprocess.run(["/bin/cp", "-cR", Path(bundle).resolve(strict=True), app], check=True, timeout=20)
        contents = app / "Contents"
        python = contents / desktop_package.PYTHON
        runtime = contents / "Resources" / "python"
        for cache in runtime.rglob("__pycache__/*.pyc"):
            if Path(source_from_cache(str(cache))).is_file():
                cache.unlink()

        def inventory():
            directories = {
                relative: stat.S_IMODE(path.stat().st_mode)
                for relative, path in seal._entries(contents)
                if not path.is_symlink() and path.is_dir()
            }
            return seal._inventory(contents, sealed=True), directories

        before = inventory()
        project = root / "cold-train"
        for name, data in dump_text_project(
            managed_train_project(batches=1), ProjectConfiguration(project="cold-train", store="results"),
        ).items():
            destination = project / name
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_bytes(data)
        host = HostClient(root / "host")
        host.prohibit_start()
        assert not host.root.exists()
        env = {
            "HOME": str(root), "PATH": os.defpath, "FEEDBAX_HOST_ROOT": str(host.root),
            "JAX_PLATFORMS": "cpu", "JAX_COMPILATION_CACHE_DIR": str(root / "jax"),
        }
        try:
            output = subprocess.check_output(
                [python, "-I", "-B", "-m", "feedbax.cli", "run", "--json", "."],
                cwd=project, env=env, timeout=18,
            )
            attempt, = json.loads(output)["items"]
            assert attempt["disposition"] == "admitted", attempt

            def terminal():
                row, = host.jobs_snapshot(job_id=attempt["job_id"], start_host=False)["jobs"]
                return row if row["outcome"] is not None else None

            completed = wait_until(terminal, timeout=27)
            assert completed["state"] == "succeeded", completed["outcome"]
            assert completed["outcome"]["completion"]["checkpoint"]["completed_updates"] == 1
            assert Path(completed["outcome"]["completion"]["checkpoint"]["checkpoint_root"]).is_relative_to(
                project / "results",
            )
        finally:
            if host.address.exists():
                os.kill(int(host.ping()["pid"]), signal.SIGTERM)
                wait_until(lambda: not host.address.exists(), timeout=6)

        after = inventory()
        additions = sorted(after[0].keys() - before[0].keys())
        assert after == before, f"{len(additions)} files added to bundled runtime, first ten: {additions[:10]}"
