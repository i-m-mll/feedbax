"""Training run support helpers."""

from collections.abc import Sequence
import functools
import inspect
import logging
import os
from pathlib import Path, PosixPath
import platform
import signal
import subprocess
from shutil import rmtree
from time import perf_counter
from types import ModuleType
from typing import Optional

import jax
import jax.numpy as jnp
from jaxtyping import Array, Float, Shaped

logger = logging.getLogger(__name__)


class Timer:
    """Context manager for timing code blocks."""

    def __init__(self):
        self.times = []

    def __enter__(self, printout=False):
        self.start_time = perf_counter()
        self.printout = printout
        return self

    def __exit__(self, *args, **kwargs):
        self.time = perf_counter() - self.start_time
        self.times.append(self.time)
        self.readout = f"Time: {self.time:.3f} seconds"
        if self.printout:
            print(self.readout)

    start = __enter__
    stop = __exit__


def delete_contents(path: str | Path):
    """Delete all subdirectories and files of `path`."""
    for p in Path(path).iterdir():
        if p.is_dir():
            rmtree(p)
        elif p.is_file():
            p.unlink()


def _dirname_of_this_module():
    """Return the directory containing this module."""
    return os.path.dirname(os.path.abspath(__file__))


def git_commit_id(
    path: Optional[str | PosixPath] = None,
    module: Optional[ModuleType] = None,
) -> str:
    """Get the checked-out commit ID in the repo at `path`."""
    if path is None:
        if module is None:
            path = _dirname_of_this_module()
        else:
            path = Path(module.__file__).absolute().parent

    return (
        subprocess.check_output(["git", "describe", "--always"], cwd=path)
        .strip()
        .decode()
    )


def discard(*args, **kwargs) -> None:
    """No-op callback that accepts and discards any arguments."""
    return None


def with_caller_logger(func):
    """Provide the caller's logger to wrapped functions that accept `logger=`."""
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        if "logger" not in kwargs:
            caller_module = None
            caller_frame = inspect.currentframe()
            if caller_frame is not None:
                caller_module = inspect.getmodule(caller_frame.f_back)
            if caller_module is not None:
                kwargs["logger"] = logging.getLogger(caller_module.__name__)
            else:
                kwargs["logger"] = logging.getLogger(func.__module__)
        return func(*args, **kwargs)

    return wrapper


@with_caller_logger
def log_version_info(
    *args: ModuleType,
    git_modules: Optional[Sequence[ModuleType]] = None,
    python_version: bool = True,
    logger: logging.Logger = logger,
    level: int = logging.DEBUG,
) -> dict[str, str]:
    version_info: dict[str, str] = {}

    log_strs = []
    if python_version:
        python_ver = platform.python_version()
        version_info["python"] = python_ver
        log_strs.append(f"python version: {python_ver}")

    for package in args:
        version = package.__version__
        version_info[package.__name__] = version
        log_strs.append(f"{package.__name__} version: {version}")

    if git_modules:
        for module in git_modules:
            commit = git_commit_id(module=module)
            version_info[f"{module.__name__} commit"] = commit
            log_strs.append(f"{module.__name__} commit: {commit}")

    for s in log_strs:
        logger.log(level, s)

    return version_info


def batched_outer(
    x: Shaped[Array, "*batch n"],
    y: Shaped[Array, "*batch n"],
) -> Shaped[Array, "*batch n n"]:
    """Return the outer product of the final dimension of an array."""
    return jnp.einsum("...i,...j->...ij", x, y)


def exponential_smoothing(
    x: Float[Array, "*"],
    alpha: float,
    init_window_size: int = 1,
    axis: int = -1,
):
    """Return the exponential moving average of an array along `axis`."""
    alpha = jnp.clip(alpha, 0, 1)
    init_value = jnp.mean(jnp.take(x, jnp.arange(init_window_size), axis=axis), axis=axis)
    x_moved = jnp.moveaxis(x, axis, 0)

    def scan_fn(carry, x_t):
        ema = (1 - alpha) * carry + alpha * x_t
        return ema, ema

    _, ema = jax.lax.scan(scan_fn, init_value, x_moved)
    return jnp.moveaxis(ema, 0, axis)


class GracefulStopRequested(Exception):
    """Custom exception for graceful stopping."""


class GracefulInterruptHandler:
    """Context manager and decorator for graceful keyboard interrupt handling."""

    def __init__(
        self,
        sensitive_msg: Optional[str] = None,
        stop_msg: Optional[str] = None,
        logger: Optional[logging.Logger] = None,
    ):
        self.stop_requested = False
        self.in_sensitive_operation = False
        self.original_handler = None
        self.sensitive_msg = (
            sensitive_msg or "Ctrl-C caught: will exit after current operation completes..."
        )
        self.stop_msg = stop_msg or "Operation completed, stopping as requested..."
        self.logger = logger

    def _log_message(self, message: str):
        if self.logger:
            self.logger.info(message)
        else:
            print(f"\n{message}")

    def __enter__(self):
        self.original_handler = signal.signal(signal.SIGINT, self._signal_handler)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        signal.signal(signal.SIGINT, self.original_handler)

    def _signal_handler(self, signum, frame):
        if self.stop_requested:
            signal.signal(signal.SIGINT, signal.SIG_DFL)
            raise KeyboardInterrupt
        self.stop_requested = True
        if self.in_sensitive_operation:
            self._log_message(self.sensitive_msg)
        else:
            self._log_message("Ctrl-C caught: stopping...")
            raise KeyboardInterrupt

    def __call__(self, func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            if self.stop_requested:
                raise GracefulStopRequested()

            self.in_sensitive_operation = True
            try:
                result = func(*args, **kwargs)
                if self.stop_requested:
                    self._log_message(self.stop_msg)
                    raise GracefulStopRequested()
                return result
            finally:
                self.in_sensitive_operation = False

        return wrapper
