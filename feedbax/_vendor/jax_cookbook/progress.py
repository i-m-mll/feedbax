# Modified by Feedbax: internal imports use the bundled package namespace.
import atexit
import logging
import math
import os
import sys
import time
from collections import deque
from collections.abc import Callable, Iterable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import timedelta
from functools import partial
from typing import Any, Dict, Optional, Tuple, TypeVar

import jax.tree as jt
from jaxtyping import PyTree
from rich import get_console
from rich.console import Console, Group, RenderableType
from rich.padding import Padding
from rich.progress import (
    BarColumn,
    # IterationSpeedColumn,
    MofNCompleteColumn,
    Progress,
    ProgressColumn,
    SpinnerColumn,
    Task,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)
from rich.text import Text
from tqdm.auto import tqdm as _tqdm

from feedbax._vendor.jax_cookbook.tree import HR

S = TypeVar("S")
T = TypeVar("T")


tqdm_mode = os.environ.get("FEEDBAX_TQDM", "auto")
_tqdm_write = partial(_tqdm.write, file=sys.stdout, end="")


# class ProgressBottom(Progress):
#     """Render tasks bottom-up: the first-added task appears at the bottom."""

#     # Rich builds the table from a list of Task objects; we just reverse that list.
#     def make_tasks_table(self, tasks: Iterable[Task]):  # type: ignore[override]
#         # Convert to list and reverse the order before delegating to parent
#         tasks_list = list(tasks)[::-1]
#         for t in tasks_list:
#             t.fields.setdefault("subdescription", "")
#         return super().make_tasks_table(tasks_list)


class _PerTaskExpDecayRate:
    """Tracks EWMA speed per task.id."""

    def __init__(self):
        # task.id -> (last_t, last_completed, ewma_rate)
        self._state: Dict[int, Tuple[float, float, float]] = {}

    def update_and_get_rate(self, task: Task) -> float | None:
        now = time.perf_counter()
        completed = float(task.completed)

        state = self._state.get(task.id)
        if state is None:
            # First sighting: seed state, no estimate yet.
            self._state[task.id] = (now, completed, 0.0)
            return None

        last_t, last_c, ewma = state
        dc = completed - last_c
        if dc <= 0:
            # No new work completed: HOLD the estimate; don't decay, don't advance time.
            return ewma if ewma > 0 else None

        dt = now - last_t
        if dt <= 1e-9:
            # Too soon to say anything meaningful.
            return ewma if ewma > 0 else None

        # Halflife → time constant.
        hl = float(task.fields.get("eta_halflife", 10.0))
        if hl <= 0:
            hl = 1e-3
        tau = hl / math.log(2.0)

        # Event-driven smoothing factor based on the inter-increment time.
        alpha = 1.0 - math.exp(-dt / tau)

        inst = max(0.0, dc / dt)
        ewma = (1.0 - alpha) * ewma + alpha * inst

        # Update the state ONLY when we actually advanced.
        self._state[task.id] = (now, completed, ewma)
        return ewma if ewma > 0 else None


class _PerTaskWindowRate:
    def __init__(self, window_s: float = 20.0):
        self.window_s = float(window_s)
        self.buf = {}  # task.id -> deque[(t, completed)]

    def update_and_get_rate(self, task: Task) -> float | None:
        now = time.perf_counter()
        c = float(task.completed)
        dq = self.buf.setdefault(task.id, deque())

        # Append only when completed changes (event samples).
        if dq and dq[-1][1] == c:
            return None if len(dq) < 2 else (dq[-1][1] - dq[0][1]) / max(1e-9, dq[-1][0] - dq[0][0])

        dq.append((now, c))

        # Trim old samples.
        cutoff = now - self.window_s
        while dq and dq[0][0] < cutoff:
            dq.popleft()

        if len(dq) < 2:
            return None
        dt = dq[-1][0] - dq[0][0]
        dc = dq[-1][1] - dq[0][1]
        return dc / dt if dt > 1e-9 and dc > 0 else None


class PerTaskSpeedColumn(ProgressColumn):
    """Shows it/s using per-task EWMA; configurable via task.fields['eta_halflife']."""

    def __init__(
        self, rates: _PerTaskExpDecayRate, style: str = "progress.spinner", parens: bool = False
    ):
        super().__init__()
        self._rates = rates
        self.style = style
        self._parens = parens

    def render(self, task: Task) -> RenderableType:
        rate = self._rates.update_and_get_rate(task)
        speed_str = f"{rate:,.2f} it/s" if rate else "--- it/s"
        if self._parens:
            speed_str = f"({speed_str})"
        return Text(speed_str, style=self.style)


ETA_PLACEHOLDER = "--:--:--"


class PerTaskETAColumn(ProgressColumn):
    """Shows ETA using per-task EWMA speed; configurable via task.fields['eta_halflife']."""

    def __init__(
        self, rates: _PerTaskExpDecayRate, style: str = "progress.remaining", label: str = ""
    ):
        super().__init__()
        self._rates = rates
        self.style = style
        self._label = label

    # @staticmethod
    # def _fmt_seconds(sec: float) -> str:
    #     if not math.isfinite(sec) or sec < 0:
    #         return "—"
    #     m, s = divmod(int(sec + 0.5), 60)
    #     h, m = divmod(m, 60)
    #     if h:
    #         return f"{h:d}:{m:02d}:{s:02d}"
    #     return f"{m:d}:{s:02d}"

    def render(self, task: Task) -> RenderableType:
        if task.total is None or task.completed >= task.total:
            eta_str = ETA_PLACEHOLDER
        assert task.total is not None
        rate = self._rates.update_and_get_rate(task)
        if not rate:
            eta_str = ETA_PLACEHOLDER
        else:
            remaining = max(0.0, float(task.total) - float(task.completed))
            eta = remaining / rate
            if rate > 0:
                eta_str = time.strftime("%H:%M:%S", time.gmtime(eta))
            else:
                eta_str = ETA_PLACEHOLDER
        if self._label:
            eta_str = f"{self._label} {eta_str}"
        return Text(eta_str, style=self.style)


_rates = _PerTaskExpDecayRate()


_COLUMNS = [
    SpinnerColumn(),
    TextColumn(
        "{task.description}",
        style="markdown.strong",
    ),
    BarColumn(),
    MofNCompleteColumn(),
    TimeElapsedColumn(),
    # TimeRemainingColumn(),
    PerTaskETAColumn(_rates),
    PerTaskSpeedColumn(_rates),
    # IterationSpeedColumn(),
    TextColumn(
        "{task.fields[subdescription]}",
        style="progress.description",
        justify="right",
    ),
]
_PROG: Optional[Progress] = None
_RC = 0  # refcount of active users


def _start_progress(*, redirect_print: bool = True, **kwargs) -> Progress:
    global _PROG
    if _PROG is None:
        _PROG = Progress(
            *_COLUMNS,
            console=get_console(),  # same console as your RichHandler
            redirect_stdout=redirect_print,
            redirect_stderr=redirect_print,
            **kwargs,
        )
        _PROG.console.print()  #! ensure a blank line before the bars (doesn't seem to work)
        _PROG.start()
    return _PROG


def _stop_progress() -> None:
    global _PROG
    if _PROG is not None:
        _PROG.stop()  # transient=True clears the area
        _PROG = None


@atexit.register
def _shutdown_progress():
    # Safety: ensure clean teardown on process exit.
    _stop_progress()


def retain_progress(
    *,
    transient: bool = True,  # clear area when last task finishes
    redirect_print: bool = True,  # keep prints/logs above bars
    refresh_per_second: float = 10,
    speed_estimate_period: float = 1.0,  # smoothing window for speed estimate
) -> Progress:
    """Acquire the shared Progress (starting it if needed)."""
    global _RC
    prog = _start_progress(
        transient=transient,
        redirect_print=redirect_print,
        refresh_per_second=refresh_per_second,
        speed_estimate_period=speed_estimate_period,
    )
    _RC += 1
    return prog


def release_progress() -> None:
    """Release a previous retain; stops/clears when the last user leaves."""
    global _RC
    _RC = max(0, _RC - 1)
    if _RC == 0:
        _stop_progress()


@contextmanager
def progress_session():
    """Keep the global Progress alive for the duration of this context."""
    retain_progress()  # increments the refcount; starts the hub if needed
    try:
        yield  # tasks can be created/removed freely inside
    finally:
        release_progress()  # decrements; stops hub at the very end (one newline)


@contextmanager
def progress_task(
    description: str,
    total: Optional[int] = None,
    *,
    completed: int = 0,
    eta_halflife: float = 10.0,  # smoothing window for speed estimate (per-task override
    **kwargs,
) -> Iterator[Callable[[int], None]]:
    """
    Create a shared task, yield an `advance(n=1)` function; task is removed automatically.
    """
    prog = retain_progress(**kwargs)
    task_id = prog.add_task(
        description, total=total, completed=completed, subdescription="", eta_halflife=eta_halflife
    )
    try:
        yield lambda n=1: prog.advance(task_id, n)
    finally:
        prog.remove_task(task_id)
        release_progress()


@dataclass
class PiterUpdate:
    _update: Callable[..., None]
    _advance: Callable[[int], None]

    # Convenience methods you can call from *inside* your loop body:
    def subdescription(self, text: object) -> None:
        self._update(subdescription=str(text))

    def description(self, text: object) -> None:
        self._update(description=str(text))

    def advance(self, n: int = 1) -> None:
        self._advance(n)

    def completed(self, n: int) -> None:
        self._update(completed=int(n))

    def total(self, n: int) -> None:
        self._update(total=int(n))


@contextmanager
def progress_piter(
    iterable: Iterable[T],
    *,
    description: str = "Working…",
    total: Optional[int] = None,
    completed: int = 0,
    auto_advance: bool = True,  # auto-advance after each yielded item
    eta_halflife: float = 10.0,  # smoothing window for speed estimate
):
    """
    Context manager yielding (iterator, update).

    Use `update.subdesc(...)` or `update.desc(...)` *inside* your loop body.
    """
    if total is None:
        try:
            total = len(iterable)  # may fail for generators
        except Exception:
            total = None

    prog = retain_progress()
    task_id = prog.add_task(
        description, total=total, completed=completed, subdescription="", eta_halflife=eta_halflife
    )
    try:

        def _update(**kwargs):
            prog.update(task_id, **kwargs)

        def _advance(n=1):
            prog.advance(task_id, n)

        updater = PiterUpdate(_update=_update, _advance=_advance)

        def _iter() -> Iterator[T]:
            i = completed
            for item in iterable:
                yield item
                if auto_advance:
                    i += 1
                    _advance(1)

        yield _iter(), updater
    finally:
        prog.remove_task(task_id)
        release_progress()


def piter(
    iterable: Iterable[T],
    *,
    description: str = "Working…",
    total: Optional[int] = None,
    completed: int = 0,
    right: Optional[Callable[[T, int], str]] = None,  # label function(item, index)
    eta_halflife: float = 10.0,  # smoothing window for speed estimate
    **kwargs,
) -> Iterator[T]:
    """
    Iterator wrapper sharing the global Progress.
    """
    if total is None:
        try:
            total = len(iterable)  # may fail (generators)
        except Exception:
            total = None

    prog = retain_progress(**kwargs)
    task_id = prog.add_task(
        description, total=total, completed=completed, subdescription="", eta_halflife=eta_halflife
    )
    try:
        for i, item in enumerate(iterable):
            if right is not None:
                try:
                    prog.update(task_id, subdescription=right(item, i))
                except Exception:
                    pass  # don't break progress on label errors
            yield item
            prog.advance(task_id, 1)
    finally:
        prog.remove_task(task_id)
        release_progress()


def map_rich(
    f: Callable[..., S],
    tree: PyTree[Any, "T"],
    *rest: PyTree[Any, "T"],
    description: str = "Processing tree leaves",
    labels: Optional[PyTree[str, "T"]] = None,
    verbose: bool = False,
    is_leaf: Optional[Callable[..., bool]] = None,
    logger: Optional[logging.Logger] = None,
    log_level: int = logging.INFO,
    eta_halflife: float = 10.0,  # smoothing window for speed estimate
    **kwargs,  # for `retain_progress`
) -> PyTree[S, "T"]:
    """Adds a Rich progress task to a shared global Progress (stacks with others)."""
    n_leaves = len(jt.leaves(tree, is_leaf=is_leaf))

    # We still grab the global console for fallback logging
    console = get_console()

    # NEW: acquire the shared Progress instead of creating a local one
    prog = retain_progress(**kwargs)
    task_id = prog.add_task(
        description, total=n_leaves, subdescription="", eta_halflife=eta_halflife
    )
    try:
        # Prepare labels tree (match your tqdm version)
        if labels is None:
            labels = jt.map(lambda _: None, tree, is_leaf=is_leaf)

        def _log(msg: str):
            if verbose:
                if logger is not None:
                    logger.log(log_level, msg)
                else:
                    console.log(msg)

        def _f(leaf, leaf_label, *rest_args):
            if leaf_label is not None:
                prog.update(task_id, subdescription=leaf_label)
            if verbose:
                _log(f"Processing leaf: {leaf_label}")

            result = f(leaf, *rest_args)

            prog.advance(task_id, 1)
            return result

        return jt.map(_f, tree, labels, *rest, is_leaf=is_leaf)
    finally:
        # Remove just this task (so it vanishes immediately)
        prog.remove_task(task_id)
        # Release the shared Progress; if this was the last user, it stops (and clears if transient)
        release_progress()


# TODO: Use a host callback so this can be wrapped in JAX transformations.
# See https://github.com/jeremiecoullon/jax-tqdm for a similar example.
# (Currently I only use this function when `f` is a `TaskTrainer`.)
def map_tqdm(
    f: Callable[..., S],
    tree: PyTree[Any, "T"],
    *rest: PyTree[Any, "T"],
    label: Optional[str] = None,
    labels: Optional[PyTree[str, "T"]] = None,
    verbose: bool = False,
    is_leaf: Optional[Callable[..., bool]] = None,
) -> PyTree[S, "T"]:
    """Adds a tqdm progress bar to `tree_map`.

    Arguments:
        f: The function to map over the tree.
        tree: The PyTree to map over.
        *rest: Additional arguments to `f`, as PyTrees with the same structure as `tree`.
        label: A single label to be displayed irrespective of the leaf being processed.
            Overridden by `labels`.
        labels: A PyTree of labels for the leaves of `tree`, to be displayed on the
            progress bar.
        is_leaf: A function that returns `True` for leaves of `tree`.
    """
    n_leaves = len(jt.leaves(tree, is_leaf=is_leaf))
    pbar = _tqdm(total=n_leaves, desc=label)

    def _f(leaf, label, *rest):
        if label is not None:
            pbar.set_description(f"Processing leaf: {label}")
        if verbose:
            _tqdm_write(f"Processing leaf: {label}\n\n")
        result = f(leaf, *rest)
        if verbose:
            _tqdm_write(f"\n{HR}\n")
        else:
            _tqdm_write("\n")
        pbar.update(1)
        return result

    if labels is None:
        pbar.set_description("Processing tree leaves")
        labels = jt.map(lambda _: None, tree, is_leaf=is_leaf)
    return jt.map(_f, tree, labels, *rest, is_leaf=is_leaf)


def display_rich_text_themes():
    """Displays all available text styles for the default `rich` console."""
    console = Console()

    # Get all style names in the current theme
    all_styles = sorted(console._theme_stack._entries[0].keys())  # private, but works

    # Print them out
    for name in all_styles:
        console.print(f"{name:25}", style=name)
