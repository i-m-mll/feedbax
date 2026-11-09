"""Install or recover a verified whole Studio release on macOS."""

from __future__ import annotations

import argparse
import fcntl
import json
import os
import shutil
import stat
import subprocess
import tempfile
from contextlib import contextmanager
from pathlib import Path, PurePosixPath
from zipfile import ZipFile

from seal import APP_NAME, LAUNCHER, json_bytes, sha256, verify

JOURNAL_SCHEMA = "feedbax.desktop.install.v1"
STAGE_PREFIX = ".feedbax-replace-"


def _service(app: Path, *args: str, lock: int | None = None) -> dict:
    arguments = [str(app / "Contents" / LAUNCHER), "service", *args]
    if lock is not None:
        arguments.extend(("--installation-fd", str(lock)))
    output = subprocess.run(
        arguments, check=True, capture_output=True, text=True, timeout=55,
        pass_fds=(lock,) if lock is not None else (),
    )
    return json.loads(output.stdout)


def _sync(directory: Path) -> None:
    fd = os.open(directory, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def _record(path: Path, document: dict) -> None:
    fd, name = tempfile.mkstemp(dir=path.parent, prefix=".installation-")
    temporary = Path(name)
    try:
        with os.fdopen(fd, "wb") as output:
            output.write(json_bytes(document))
            output.flush()
            os.fsync(output.fileno())
        temporary.replace(path)
        _sync(path.parent)
    finally:
        temporary.unlink(missing_ok=True)


@contextmanager
def _installation(control: Path):
    lock = os.open(control / "install.lock", os.O_CREAT | os.O_RDWR | os.O_NOFOLLOW, 0o600)
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        yield lock
    finally:
        os.close(lock)


def _stage_archive(archive: Path, expected: dict, directory: Path) -> Path:
    if archive.stat().st_size != expected["archive"]["bytes"] or sha256(archive) != expected["archive"]["sha256"]:
        raise ValueError("distribution archive does not match its receipt")
    with ZipFile(archive) as container:
        for item in container.infolist():
            path = PurePosixPath(item.filename)
            if path.is_absolute() or ".." in path.parts or path.parts[0] not in (APP_NAME, "__MACOSX"):
                raise ValueError(f"unexpected distribution archive member {item.filename!r}")
            if stat.S_ISLNK(item.external_attr >> 16):
                link = PurePosixPath(os.fsdecode(container.read(item)))
                if link.is_absolute():
                    raise ValueError(f"absolute link in distribution archive: {item.filename}")
                depth = len(path.parts) - 1
                for part in link.parts:
                    depth += -1 if part == ".." else 1
                    if depth < 1:
                        raise ValueError(f"escaping link in distribution archive: {item.filename}")
    subprocess.run(["/usr/bin/ditto", "-x", "-k", archive, directory], check=True, timeout=55)
    app = directory / APP_NAME
    if verify(app) != {name: value for name, value in expected.items() if name != "archive"}:
        raise ValueError("unpacked app does not match its sealed receipt")
    return app


def _stop(app: Path, disposition: str, lock: int) -> dict | None:
    status = _service(app, "health")["status"]
    if status == "ready":
        outcome = _service(app, "stop", disposition, lock=lock)
        if outcome["status"] != "ready_to_stop":
            return outcome
    elif status != "unavailable":
        raise RuntimeError(f"cannot establish safe Studio stop: {status}")
    _service(app, "unregister", lock=lock)
    return None


def _activate(app: Path, receipt: dict, lock: int) -> dict:
    registration = _service(app, "register", lock=lock)["status"]
    if registration != "enabled":
        raise RuntimeError(f"Studio background service activation returned {registration}")
    health = _service(app, "health", "--wait", "20")
    if (
        health["status"] != "ready"
        or health["release"] != receipt["manifest_sha256"]
        or health["installed_path"] != str(app / "Contents")
        or health["runtime"]["source_commit"] != receipt["source_commit"]
    ):
        raise RuntimeError(f"the selected Studio release is not ready: {health}")
    return health


def _require_closed_gui(app: Path) -> None:
    executable = str(app / "Contents" / LAUNCHER)
    processes = subprocess.check_output(
        ["/bin/ps", "-axww", "-o", "command="], text=True, timeout=5,
    )
    if executable in processes.splitlines():
        raise RuntimeError("Quit the installed Studio GUI before replacing or removing it")


def _same_line(old: dict, new: dict) -> bool:
    return old["bundle_id"] == new["bundle_id"] and old["signing"]["team"] == new["signing"]["team"]


def _transaction_paths(document: dict, destination: Path) -> tuple[Path, Path, Path]:
    stage = Path(document["stage"])
    if (
        document.get("schema") != JOURNAL_SCHEMA
        or document["destination"] != str(destination)
        or stage.parent != destination.parent
        or not stage.name.startswith(STAGE_PREFIX)
        or stage.is_symlink()
    ):
        raise ValueError("installation journal does not belong to the requested destination")
    return stage, stage / APP_NAME, stage / "previous.app"


def _finish(journal: Path, document: dict, destination: Path) -> None:
    stage, _, _ = _transaction_paths(document, destination)
    _record(journal, document | {"phase": "complete"})
    if stage.exists():
        shutil.rmtree(stage)
        _sync(stage.parent)
    journal.unlink()
    _sync(journal.parent)


def _replace(journal: Path, document: dict, destination: Path, lock: int, *, prepared: bool = False) -> dict:
    _, candidate, previous = _transaction_paths(document, destination)
    old, new = document["old"], document["new"]
    installed = old
    if not prepared:
        installed = verify(destination) if destination.exists() else None
    if installed != new:
        if installed != old and installed is not None:
            raise RuntimeError("the destination contains an unexpected signed release")
        if old is not None and installed == old:
            _require_closed_gui(destination)
            blocked = _stop(destination, document["disposition"], lock)
            if blocked is not None:
                return blocked
            destination.rename(previous)
            _sync(destination.parent)
            _sync(previous.parent)
        if not prepared and (verify(candidate) != new or (old is not None and verify(previous) != old)):
            raise RuntimeError("staged or rollback bytes changed during installation")
        candidate.rename(destination)
        _sync(candidate.parent)
        _sync(destination.parent)
    health = _activate(destination, new, lock)
    _finish(journal, document, destination)
    return {"status": "installed", "release": new, "service": health}


def install(archive: Path, receipt_path: Path, destination: Path, disposition: str) -> dict:
    destination.parent.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix=STAGE_PREFIX, dir=destination.parent))
    recorded = False
    try:
        expected = json.loads(receipt_path.read_bytes())
        candidate = _stage_archive(archive, expected, stage)
        new = {name: value for name, value in expected.items() if name != "archive"}
        control = Path(_service(candidate, "paths")["control"])
        journal = control / "installation.json"
        with _installation(control) as lock:
            if journal.exists():
                raise RuntimeError("an interrupted installation exists; run recover")
            old = verify(destination) if destination.exists() else None
            if old is not None and (
                not _same_line(old, new) or _service(destination, "paths")["control"] != str(control)
            ):
                raise ValueError("replacement belongs to a different signing or installation identity")
            document = {
                "schema": JOURNAL_SCHEMA, "phase": "prepared", "destination": str(destination),
                "stage": str(stage), "old": old, "new": new, "disposition": disposition,
            }
            _record(journal, document)
            recorded = True
            outcome = _replace(journal, document, destination, lock, prepared=True)
            if outcome["status"] == "deferred":
                _finish(journal, document, destination)
            return outcome
    finally:
        if not recorded and stage.exists():
            shutil.rmtree(stage)


def recover(control: Path, destination: Path, rollback: bool) -> dict:
    journal = control / "installation.json"
    with _installation(control) as lock:
        document = json.loads(journal.read_bytes())
        stage, candidate, previous = _transaction_paths(document, destination)
        if document["phase"] == "complete":
            _finish(journal, document, destination)
            return {"status": "cleanup_complete"}
        if not rollback:
            return _replace(journal, document, destination, lock)
        old, new = document["old"], document["new"]
        installed = verify(destination) if destination.exists() else None
        if installed not in (old, new, None):
            raise RuntimeError("cannot roll back an unexpected installed release")
        if installed == new:
            _require_closed_gui(destination)
            blocked = _stop(destination, document["disposition"], lock)
            if blocked is not None:
                return blocked
            destination.rename(candidate)
            _sync(destination.parent)
            _sync(stage)
        if old is not None and installed != old:
            if verify(previous) != old:
                raise RuntimeError("rollback release does not match the journal")
            previous.rename(destination)
            _sync(destination.parent)
            _sync(stage)
        health = _activate(destination, old, lock) if old is not None else None
        _finish(journal, document, destination)
        return {"status": "rolled_back", "service": health}


def uninstall(destination: Path, disposition: str) -> dict:
    verify(destination)
    control = Path(_service(destination, "paths")["control"])
    with _installation(control) as lock:
        if (control / "installation.json").exists():
            raise RuntimeError("recover the interrupted installation before uninstalling")
        _require_closed_gui(destination)
        blocked = _stop(destination, disposition, lock)
        if blocked is not None:
            return blocked
        with tempfile.TemporaryDirectory(prefix=STAGE_PREFIX, dir=destination.parent) as stage:
            destination.rename(Path(stage) / APP_NAME)
            _sync(destination.parent)
        return {"status": "uninstalled", "data": "preserved"}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="action", required=True)
    for action in ("install", "recover", "uninstall"):
        command = sub.add_parser(action)
        command.add_argument("destination", type=Path)
        if action == "recover":
            command.add_argument("--control", type=Path, required=True)
            command.add_argument("--rollback", action="store_true")
        else:
            command.add_argument("--jobs", choices=("defer", "checkpoint", "force"), default="defer")
        if action == "install":
            command.add_argument("--archive", type=Path, required=True)
            command.add_argument("--receipt", type=Path, required=True)
    args = parser.parse_args()
    if args.destination.is_symlink() or args.destination.suffix != ".app":
        parser.error("destination must be a real .app bundle path")
    destination = args.destination.resolve()
    if args.action == "install":
        result = install(args.archive, args.receipt, destination, args.jobs)
    elif args.action == "recover":
        result = recover(args.control.resolve(strict=True), destination, args.rollback)
    else:
        result = uninstall(destination, args.jobs)
    print(json.dumps(result))


if __name__ == "__main__":
    main()
