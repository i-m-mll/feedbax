from concurrent.futures import ThreadPoolExecutor
import errno
import os
from pathlib import Path
import stat
from threading import Barrier

import pytest

from feedbax import _atomic
from feedbax.analysis import evaluation_compaction, evaluation_product_union


@pytest.mark.parametrize(
    "writer",
    [
        evaluation_compaction._atomic_write_json,
        evaluation_product_union._atomic_write_json,
    ],
)
def test_overlapping_evaluation_publishers_keep_complete_distinct_documents(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    writer,
) -> None:
    destination = tmp_path / "receipt.json"
    destination.write_bytes(b"old\n")
    expected = {
        b'{\n  "author": "one",\n  "value": "\\u00e9"\n}\n',
        b'{\n  "author": "two",\n  "value": "\\u00e9"\n}\n',
    }
    barrier = Barrier(2, timeout=5)
    replace = os.replace
    candidates = []

    def overlap(source, target):
        candidates.append(Path(source).read_bytes())
        assert destination.read_bytes() == b"old\n"
        barrier.wait()
        replace(source, target)

    monkeypatch.setattr(_atomic.os, "replace", overlap)
    # Exercise the same overlap against the former Path.replace implementation.
    monkeypatch.setattr(Path, "replace", overlap)
    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = [
            pool.submit(writer, destination, {"author": author, "value": "é"})
            for author in ("one", "two")
        ]
        for future in futures:
            future.result(timeout=10)
    assert set(candidates) == expected
    assert destination.read_bytes() in expected
    assert list(tmp_path.iterdir()) == [destination]


@pytest.mark.parametrize("boundary", ["fdopen", "file_sync", "replace", "directory_sync"])
def test_publication_failure_propagates_and_leaves_no_staging(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    boundary: str,
) -> None:
    destination = tmp_path / "record"
    destination.write_bytes(b"old")
    fsync = os.fsync

    def fail(*args, **kwargs):
        raise OSError(errno.EIO, "injected publication failure")

    def sync(descriptor):
        is_directory = stat.S_ISDIR(os.fstat(descriptor).st_mode)
        if is_directory == (boundary == "directory_sync"):
            fail()
        fsync(descriptor)

    if boundary == "fdopen":
        monkeypatch.setattr(_atomic.os, "fdopen", fail)
    elif boundary == "replace":
        monkeypatch.setattr(_atomic.os, "replace", fail)
    else:
        monkeypatch.setattr(_atomic.os, "fsync", sync)
    with pytest.raises(OSError, match="injected publication failure"):
        _atomic.write_bytes_atomic(destination, b"new\x00\xff\n")
    assert destination.read_bytes() == (
        b"new\x00\xff\n" if boundary == "directory_sync" else b"old"
    )
    assert list(tmp_path.iterdir()) == [destination]


def test_publication_flushes_exact_bytes_before_replace_and_syncs_directory(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    destination = tmp_path / "record"
    data = b"\x00\xff\r\n"
    fsync, replace = os.fsync, os.replace
    events = []

    def sync(descriptor):
        if stat.S_ISDIR(os.fstat(descriptor).st_mode):
            assert destination.read_bytes() == data
            events.append("directory")
        else:
            assert next(tmp_path.iterdir()).read_bytes() == data
            events.append("file")
        fsync(descriptor)

    def publish(source, target):
        assert events == ["file"]
        events.append("replace")
        replace(source, target)

    monkeypatch.setattr(_atomic.os, "fsync", sync)
    monkeypatch.setattr(_atomic.os, "replace", publish)
    _atomic.write_bytes_atomic(destination, data)
    assert events == ["file", "replace", *(["directory"] * len(destination.resolve().parents))]


def test_publication_preserves_caller_permission_policy(tmp_path: Path) -> None:
    ordinary = tmp_path / "ordinary"
    ordinary.write_bytes(b"existing")
    expected_mode = stat.S_IMODE(ordinary.stat().st_mode)
    _atomic.write_bytes_atomic(ordinary, b"replacement", mode=0o666)
    assert stat.S_IMODE(ordinary.stat().st_mode) == expected_mode
    private = tmp_path / "private"
    _atomic.write_bytes_atomic(private, b"private")
    assert stat.S_IMODE(private.stat().st_mode) == expected_mode & 0o600


@pytest.mark.parametrize("error_number", [errno.EINVAL, errno.ENOTSUP])
def test_unsupported_directory_sync_cannot_report_durable_success(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    error_number: int,
) -> None:
    destination = tmp_path / "record"
    fsync = os.fsync

    def sync(descriptor):
        if stat.S_ISDIR(os.fstat(descriptor).st_mode):
            raise OSError(error_number, "directory sync unsupported")
        fsync(descriptor)

    monkeypatch.setattr(_atomic.os, "fsync", sync)
    with pytest.raises(OSError, match="directory sync unsupported"):
        _atomic.write_bytes_atomic(destination, b"visible but unconfirmed")
    assert destination.read_bytes() == b"visible but unconfirmed"
    assert list(tmp_path.iterdir()) == [destination]


def test_nested_publication_resyncs_namespace_ancestors_on_retry(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    destination = tmp_path / "new-parent" / "new-child" / "record"
    fsync = os.fsync
    synced = []
    fail_ancestor = True
    ancestors = tuple(destination.parent.resolve().parents)

    def sync(descriptor):
        nonlocal fail_ancestor
        if stat.S_ISDIR(os.fstat(descriptor).st_mode):
            observed = os.fstat(descriptor)
            directory = next(
                path
                for path in (destination.parent.resolve(), *ancestors)
                if path.stat().st_ino == observed.st_ino and path.stat().st_dev == observed.st_dev
            )
            synced.append(directory)
            if directory == tmp_path.resolve() and fail_ancestor:
                raise OSError(errno.EIO, "ancestor sync failed")
        fsync(descriptor)

    monkeypatch.setattr(_atomic.os, "fsync", sync)
    for _ in range(2):
        with pytest.raises(OSError, match="ancestor sync failed"):
            _atomic.write_bytes_atomic(destination, b"candidate")
        assert destination.read_bytes() == b"candidate"
        assert not list(tmp_path.rglob("*.tmp"))
    synced.clear()
    fail_ancestor = False
    _atomic.write_bytes_atomic(destination, b"durable")
    assert synced == [destination.parent.resolve(), *ancestors]
    assert destination.read_bytes() == b"durable"
