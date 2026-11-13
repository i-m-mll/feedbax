"""Exercise installation recovery with exact paths and an isolated registration ledger."""

import importlib.util
from pathlib import Path

import pytest


@pytest.fixture
def installation(tmp_path, monkeypatch):
    desktop = Path(__file__).resolve().parents[1] / "desktop"
    monkeypatch.syspath_prepend(str(desktop))
    spec = importlib.util.spec_from_file_location("studio_install", desktop / "install.py")
    installer = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(installer)
    destination = tmp_path / installer.APP_NAME
    control = tmp_path / "control"
    control.mkdir()
    unrelated = tmp_path / "Other.app"
    registrations = {unrelated}
    service = {"running": False, "registered": False}

    def launch_services(app, *, register):
        assert app.is_dir()
        if register:
            assert app == destination
            registrations.add(app)
        else:
            registrations.discard(app)

    def command(app, action, *_args, lock=None):
        if action == "paths":
            return {"control": str(control)}
        assert app == destination
        if action == "register":
            service.update(running=True, registered=True)
            return {"status": "enabled"}
        if action == "stop":
            assert lock is not None
            service["running"] = False
            return {"status": "ready_to_stop"}
        if action == "unregister":
            assert not service["running"]
            service["registered"] = False
            return {"status": "not_registered"}
        if action == "health":
            current = installer.verify(app)
            return {
                "status": "ready" if service["running"] else "unavailable",
                "release": current["manifest_sha256"],
                "installed_path": str(destination / "Contents"),
                "runtime": {"source_commit": current["source_commit"]},
            }
        pytest.fail(f"unexpected service action {action}")

    def stage(_archive, expected, directory):
        app = directory / installer.APP_NAME
        app.mkdir()
        (app / "verify.json").write_bytes(installer.json_bytes({
            key: value for key, value in expected.items() if key != "archive"
        }))
        registrations.add(app)
        return app

    def verify(app):
        return installer.json.loads((app / "verify.json").read_bytes())

    process_guard = installer._require_no_bundle_host
    monkeypatch.setattr(installer, "_launch_services", launch_services)
    monkeypatch.setattr(installer, "_service", command)
    monkeypatch.setattr(installer, "_stage_archive", stage)
    monkeypatch.setattr(installer, "_require_closed_gui", lambda app: None)
    monkeypatch.setattr(installer, "_require_no_bundle_host", lambda app: None)
    monkeypatch.setattr(installer, "verify", verify)
    return installer, destination, control, unrelated, registrations, service, process_guard


def test_install_replacement_and_uninstall_only_touch_owned_registrations(installation, tmp_path):
    installer, app, control, unrelated, registrations, service, _ = installation
    receipt = tmp_path / "receipt.json"
    for revision in ("first", "second"):
        receipt.write_bytes(installer.json_bytes({
            "bundle_id": "same.bundle", "signing": {"team": "same"}, "archive": {},
            "manifest_sha256": revision, "source_commit": revision,
        }))
        assert installer.install(receipt, receipt, app, "checkpoint")["status"] == "installed"
        assert registrations == {app, unrelated}
        assert service == {"running": True, "registered": True}
        assert not (control / "installation.json").exists()
    service["running"] = False
    assert installer.uninstall(app, "checkpoint") == {"status": "uninstalled", "data": "preserved"}
    assert service == {"running": False, "registered": False}
    assert registrations == {unrelated}
    assert not app.exists() and control.is_dir()


def test_failed_uninstall_registration_is_recoverable(installation, tmp_path, monkeypatch):
    installer, app, control, unrelated, registrations, service, _ = installation
    receipt = tmp_path / "receipt.json"
    receipt.write_bytes(installer.json_bytes({
        "bundle_id": "same.bundle", "signing": {"team": "same"}, "archive": {},
        "manifest_sha256": "first", "source_commit": "first",
    }))
    installer.install(receipt, receipt, app, "defer")
    update = installer._launch_services

    def fail_after_move(path, *, register):
        if path.parent.name.startswith(installer.STAGE_PREFIX):
            registrations.add(path)
            raise OSError("transient LaunchServices failure")
        update(path, register=register)

    with monkeypatch.context() as injected:
        injected.setattr(installer, "_launch_services", fail_after_move)
        with pytest.raises(OSError, match="transient LaunchServices failure"):
            installer.uninstall(app, "checkpoint")
    assert registrations - {unrelated}
    assert not app.exists() and (control / "installation.json").exists()
    assert installer.recover(control, app, rollback=False)["status"] == "uninstalled"
    assert registrations == {unrelated}
    assert service == {"running": False, "registered": False}
    assert not (control / "installation.json").exists()


def test_old_launcher_acknowledgement_cannot_hide_a_live_bundle_host(installation, tmp_path, monkeypatch):
    installer, app, control, unrelated, registrations, _, process_guard = installation
    receipt = tmp_path / "receipt.json"
    receipt.write_bytes(installer.json_bytes({
        "bundle_id": "same.bundle", "signing": {"team": "same"}, "archive": {},
        "manifest_sha256": "first", "source_commit": "first",
    }))
    installer.install(receipt, receipt, app, "defer")

    python = app / "Contents" / "Resources" / "Python Runtime" / "bin" / "python"
    python.parent.mkdir(parents=True)
    python.touch()
    (app / "Contents" / "Resources" / "release.json").write_bytes(installer.json_bytes({
        "resources": {"python": "Resources/Python Runtime/bin/python"},
    }))
    monkeypatch.setattr(installer, "_require_no_bundle_host", process_guard)
    monkeypatch.setattr(installer.subprocess, "check_output", lambda *_args, **_kwargs: (
        f"/other/python -m feedbax.host --root /other/host\n{python} -m feedbax.host --root /old/host\n"
    ))
    with pytest.raises(RuntimeError, match="still owns a Host"):
        installer.uninstall(app, "force")
    assert app.exists() and registrations == {app, unrelated}
    assert not (control / "installation.json").exists()
