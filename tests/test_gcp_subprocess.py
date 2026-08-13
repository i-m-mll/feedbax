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
        block_cleanup_until_killed: bool = False,
        block_cleanup_until_released: bool = False,
        cleanup_error: BaseException | None = None,
        initial_error: BaseException | None = None,
    ) -> None:
        self.stdout = stdout
        self.stderr = stderr
        self.returncode = returncode
        self.block_initial_communicate = block_initial_communicate
        self.block_cleanup_until_killed = block_cleanup_until_killed
        self.block_cleanup_until_released = block_cleanup_until_released
        self.cleanup_error = cleanup_error
        self.initial_error = initial_error
        self.communicate_started = asyncio.Event()
        self.cleanup_started = asyncio.Event()
        self.release_cleanup = asyncio.Event()
        self.killed = asyncio.Event()
        self.events: list[str] = []
        self.communicate_calls = 0
        self.terminate_calls = 0
        self.kill_calls = 0

    async def communicate(self) -> tuple[bytes, bytes]:
        self.communicate_calls += 1
        if self.communicate_calls == 1 and self.block_initial_communicate:
            self.communicate_started.set()
            await asyncio.Future()
        if self.communicate_calls == 1 and self.initial_error is not None:
            raise self.initial_error
        if self.communicate_calls > 1:
            self.events.append("drain")
            self.cleanup_started.set()
            if self.block_cleanup_until_killed:
                await self.killed.wait()
            elif not self.block_cleanup_until_released:
                self.release_cleanup.set()
            await self.release_cleanup.wait()
            if self.cleanup_error is not None:
                raise self.cleanup_error
            if self.returncode is None:
                self.returncode = -15
        return self.stdout, self.stderr

    def terminate(self) -> None:
        self.terminate_calls += 1
        self.events.append("terminate")

    def kill(self) -> None:
        self.kill_calls += 1
        self.events.append("kill")
        self.returncode = -9
        self.killed.set()
        self.release_cleanup.set()


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
        assert process.communicate_calls == 2
        assert process.events == ["terminate", "drain"]
        assert process.returncode == -15

    asyncio.run(run())


def test_cancelled_gcloud_process_escalates_to_kill(monkeypatch):
    async def run() -> None:
        process = _FakeProcess(
            returncode=None,
            block_initial_communicate=True,
            block_cleanup_until_killed=True,
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
        assert process.communicate_calls == 2
        assert process.events == ["terminate", "drain", "kill"]
        assert process.returncode == -9

    asyncio.run(run())


def test_repeated_cancellation_cannot_interrupt_cleanup(monkeypatch):
    async def run() -> None:
        process = _FakeProcess(
            returncode=None,
            block_initial_communicate=True,
            block_cleanup_until_released=True,
        )
        _install_process(monkeypatch, process)

        task = asyncio.create_task(gcp._run_gcloud("compute"))
        await process.communicate_started.wait()
        task.cancel()
        await process.cleanup_started.wait()
        task.cancel()
        await asyncio.sleep(0)

        assert not task.done()
        process.release_cleanup.set()
        with pytest.raises(asyncio.CancelledError):
            await task

        assert process.communicate_calls == 2
        assert process.returncode == -15

    asyncio.run(run())


def test_cleanup_failure_does_not_replace_original_error(monkeypatch):
    async def run() -> None:
        original = RuntimeError("original communication failure")
        process = _FakeProcess(
            returncode=None,
            initial_error=original,
            cleanup_error=ValueError("cleanup failed"),
        )
        _install_process(monkeypatch, process)

        with pytest.raises(RuntimeError, match="original communication failure") as raised:
            await gcp._run_gcloud("compute")

        assert raised.value is original
        assert process.communicate_calls == 2
        assert process.terminate_calls == 1

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
        assert process.communicate_calls == 1

    asyncio.run(run())
