import asyncio
import json

import pytest

from feedbax.web.orchestration import gcp


class _FakeProcess:
    def __init__(
        self,
        *,
        stdout: bytes = b"{}",
        stderr: bytes = b"",
        returncode: int | None = 0,
        block_initial_communicate: bool = False,
        block_terminated_wait: bool = False,
    ) -> None:
        self.stdout = stdout
        self.stderr = stderr
        self.returncode = returncode
        self.block_initial_communicate = block_initial_communicate
        self.block_terminated_wait = block_terminated_wait
        self.communicate_started = asyncio.Event()
        self.communicate_calls = 0
        self.wait_calls = 0
        self.terminate_calls = 0
        self.kill_calls = 0

    async def communicate(self) -> tuple[bytes, bytes]:
        self.communicate_calls += 1
        if self.communicate_calls == 1 and self.block_initial_communicate:
            self.communicate_started.set()
            await asyncio.Future()
        return self.stdout, self.stderr

    def terminate(self) -> None:
        self.terminate_calls += 1

    def kill(self) -> None:
        self.kill_calls += 1
        self.returncode = -9

    async def wait(self) -> int:
        self.wait_calls += 1
        if self.wait_calls == 1 and self.block_terminated_wait:
            await asyncio.Future()
        if self.returncode is None:
            self.returncode = -15
        return self.returncode


def _install_process(monkeypatch: pytest.MonkeyPatch, process: _FakeProcess) -> None:
    async def create_subprocess_exec(*args: str, **kwargs: object) -> _FakeProcess:
        return process

    monkeypatch.setattr(asyncio, "create_subprocess_exec", create_subprocess_exec)


def test_cancelled_gcloud_process_is_terminated_and_reaped(monkeypatch):
    async def run() -> None:
        process = _FakeProcess(returncode=None, block_initial_communicate=True)
        _install_process(monkeypatch, process)

        task = asyncio.create_task(gcp._run_gcloud("compute", "instances", "list"))
        await process.communicate_started.wait()
        task.cancel()

        with pytest.raises(asyncio.CancelledError):
            await task

        assert process.terminate_calls == 1
        assert process.kill_calls == 0
        assert process.wait_calls == 1
        assert process.communicate_calls == 2
        assert process.returncode == -15

    asyncio.run(run())


def test_cancelled_gcloud_process_escalates_to_kill(monkeypatch):
    async def run() -> None:
        process = _FakeProcess(
            returncode=None,
            block_initial_communicate=True,
            block_terminated_wait=True,
        )
        _install_process(monkeypatch, process)
        monkeypatch.setattr(gcp, "_GCLOUD_TERMINATION_GRACE_SECONDS", 0.01)

        task = asyncio.create_task(gcp._run_gcloud("compute", "instances", "list"))
        await process.communicate_started.wait()
        task.cancel()

        with pytest.raises(asyncio.CancelledError):
            await task

        assert process.terminate_calls == 1
        assert process.kill_calls == 1
        assert process.wait_calls == 2
        assert process.communicate_calls == 2
        assert process.returncode == -9

    asyncio.run(run())


@pytest.mark.parametrize(
    ("process", "expected", "error_type", "error_match"),
    [
        (_FakeProcess(stdout=b'{"status": "RUNNING"}'), {"status": "RUNNING"}, None, None),
        (
            _FakeProcess(stderr=b"provider failed", returncode=7),
            None,
            RuntimeError,
            r"gcloud compute failed \(exit 7\):\nprovider failed",
        ),
        (_FakeProcess(stdout=b"not-json"), None, json.JSONDecodeError, None),
    ],
)
def test_gcloud_completion_behavior_is_unchanged(
    monkeypatch,
    process,
    expected,
    error_type,
    error_match,
):
    async def run() -> None:
        _install_process(monkeypatch, process)
        if error_type is None:
            assert await gcp._run_gcloud("compute") == expected
        else:
            with pytest.raises(error_type, match=error_match):
                await gcp._run_gcloud("compute")

        assert process.terminate_calls == 0
        assert process.kill_calls == 0
        assert process.wait_calls == 0
        assert process.communicate_calls == 1

    asyncio.run(run())
