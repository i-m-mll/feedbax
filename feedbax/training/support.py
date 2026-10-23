"""Training run support helpers."""

from collections.abc import Sequence
import functools
import inspect
import logging
import os
from pathlib import Path, PosixPath
import platform
import subprocess
from types import ModuleType
from typing import Optional

import jax
import jax.numpy as jnp
from jaxtyping import Array, Float, Shaped

logger = logging.getLogger(__name__)


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
