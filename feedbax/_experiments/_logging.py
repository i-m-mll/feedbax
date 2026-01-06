import logging
import logging.handlers
import re
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from logging.handlers import RotatingFileHandler
from pathlib import Path
from queue import Queue
from typing import Literal, Optional

import jax.tree as jt
from rich.highlighter import ReprHighlighter
from rich.logging import RichHandler
from rich.text import Text

from feedbax._experiments.config import LOGGING, PATHS
from feedbax._experiments.types import TreeNamespace

SESSION_START_BANNER = "―" * 20 + " NEW SESSION STARTED " + "―" * 20


def wire_queue(
    *,
    queue: Queue,
    bootstrap_handler: logging.Handler,
    mode: Literal["wire", "flush"] = "flush",
    respect_handler_level: bool = True,
) -> logging.handlers.QueueListener | None:
    """Connect a QueueHandler's queue to the current root handlers.
    - mode='flush': drain once and detach the bootstrap handler.
    - mode='wire' : start a QueueListener (steady-state) and keep queue-based logging.
    """
    root = logging.getLogger()
    # Final handlers are whatever enable_logging_handlers attached:
    final_handlers = [h for h in root.handlers if h is not bootstrap_handler]

    if mode == "flush":
        listener = logging.handlers.QueueListener(
            queue, *final_handlers, respect_handler_level=respect_handler_level
        )
        listener.start()
        listener.stop()  # drains synchronously
        root.removeHandler(bootstrap_handler)
        return None

    # "wire"
    listener = logging.handlers.QueueListener(
        queue, *final_handlers, respect_handler_level=respect_handler_level
    )
    for h in final_handlers:
        root.removeHandler(h)
    listener.start()
    return listener


def _remove_handlers(logger: logging.Logger, *, predicate) -> None:
    """Remove and close all handlers on `logger` for which `predicate(handler)` is True."""
    for h in list(logger.handlers):
        if predicate(h):
            logger.removeHandler(h)
            h.close()


def _make_rotating_handler(path: Path, level: int, fmt: logging.Formatter) -> RotatingFileHandler:
    """Create a RotatingFileHandler writing to `path` at `level` with `fmt`."""
    fh = RotatingFileHandler(
        filename=str(path),
        maxBytes=LOGGING.max_bytes,
        backupCount=LOGGING.backup_count,
        encoding="utf-8",
    )
    fh.setLevel(level)
    fh.setFormatter(fmt)
    return fh


def _prune_foreign_file_handlers(central_dir: Path) -> None:
    """
    Strip any RotatingFileHandlers from *all* loggers whose baseFilename
    does not live under central_dir.
    """
    for lg in logging.Logger.manager.loggerDict.values():
        if not isinstance(lg, logging.Logger):
            continue

        for h in [h for h in lg.handlers if isinstance(h, RotatingFileHandler)]:
            try:
                # if this handler writes *outside* of our central_dir, drop it
                if Path(h.baseFilename).resolve().parent != central_dir:
                    lg.removeHandler(h)
                    h.close()
            except Exception:
                # ignore weird cases (e.g. missing baseFilename attr)
                pass


def _console_handler_pred(h: logging.Handler) -> bool:
    return isinstance(h, logging.StreamHandler) and not isinstance(h, logging.FileHandler)


#! TODO: Fix paren delimiting
class BacktickPathHighlighter(ReprHighlighter):
    # Detect a delimited chunk: "…", '…', `…`, or (…)
    _DELIM = re.compile(r"`(?P<body>[^`]+)`")
    # What "looks like a path" inside the delimiter
    _PATH = re.compile(r"^(?:~|/|[A-Za-z]:\\)[\w.\- /\\]+$")

    def highlight(self, text: Text) -> None:
        # Run the normal rules first (numbers, bools, etc.)
        super().highlight(text)

        s = text.plain
        for m in self._DELIM.finditer(s):
            # locate the inner body (whichever matched)
            body = m.group("body").strip()
            if self._PATH.match(body):
                # style only the inner content (no bleed)
                text.stylize("repr.path", m.start("body"), m.end("body"))


class _DropFileOnlyOnConsole(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        return not getattr(record, "file_only", False)


def enable_logging_handlers(
    file_level: int | None = None,
    console_level: int | None = None,
    pkg_console_levels: dict[str, int] | None = None,
    pkgs_own_files: dict[str, int] | None = None,
    *,
    # NEW: wire the bootstrap queue inside this function
    early_queue: Optional[Queue] = None,
    early_handler: Optional[logging.Handler] = None,
    queue_mode: Literal["flush", "wire"] = "flush",
    announce: bool = True,
) -> Optional[logging.handlers.QueueListener]:
    """
    Configure Rich console + file logging. If (early_queue, early_handler) are provided,
    handle the bootstrap QueueHandler here:

      queue_mode="flush": drain queued records into final handlers once, then detach early_handler.
      queue_mode="wire" : keep early_handler on root and drive final handlers via a QueueListener.

    We intentionally avoid logging until the queue is handled to prevent duplicates.
    Returns a live QueueListener in 'wire' mode, else None.
    """
    # ── 0) resolve config/defaults
    file_lvl: int = file_level or LOGGING.file_level
    console_lvl: int = console_level or LOGGING.console_level
    pkg_console_lvls: dict[str, int] = pkg_console_levels or (LOGGING.pkg_console_levels or {})
    pkg_own_fs: dict[str, int] = pkgs_own_files or (LOGGING.pkgs_own_files or {})

    console_fmt = logging.Formatter(LOGGING.console_format_str)
    file_fmt = logging.Formatter(LOGGING.file_format_str)

    # ── 1) prep central dir & prune stray file handlers
    logs_dir = Path(PATHS.logs).resolve()
    logs_dir.mkdir(parents=True, exist_ok=True)
    _prune_foreign_file_handlers(logs_dir)

    root = logging.getLogger()
    root.setLevel(1)
    root_log = logs_dir / "root.log"

    # Helper to conditionally drop console banners for package loggers
    def _log_banner(lg: logging.Logger, *, file_only: bool = False) -> None:
        extra = {"file_only": True} if file_only else None
        if lg.level > logging.INFO:
            lg.log(lg.level, SESSION_START_BANNER, extra=extra)
        else:
            lg.info(SESSION_START_BANNER, extra=extra)

    # ── 2) if flushing: detach the bootstrap handler *before* attaching finals
    #     (avoids a window where both queues and finals receive logs)
    if early_queue is not None and early_handler is not None and queue_mode == "flush":
        try:
            root.removeHandler(early_handler)
        except Exception:
            pass

    # ── 3) attach final handlers (do NOT log yet)
    # root file
    root.addHandler(_make_rotating_handler(root_log, file_lvl, file_fmt))

    # root console (Rich) with filter that drops file_only messages
    _remove_handlers(root, predicate=_console_handler_pred)
    console_h = RichHandler(level=console_lvl, highlighter=BacktickPathHighlighter())
    console_h.setFormatter(console_fmt)
    console_h.addFilter(_DropFileOnlyOnConsole())
    root.addHandler(console_h)

    # per-package console overrides (propagate to root for files)
    for pkg, lvl in pkg_console_lvls.items():
        lg = logging.getLogger(pkg)
        _remove_handlers(lg, predicate=_console_handler_pred)
        sh = RichHandler(level=lvl, highlighter=BacktickPathHighlighter())
        sh.setFormatter(console_fmt)
        sh.addFilter(_DropFileOnlyOnConsole())
        lg.addHandler(sh)
        lg.propagate = True

    # per-package own-file handlers (isolate to their own file)
    for pkg, lvl in pkg_own_fs.items():
        lg = logging.getLogger(pkg)
        _remove_handlers(
            lg, predicate=lambda h: isinstance(h, logging.handlers.RotatingFileHandler)
        )
        pkg_log = logs_dir / f"{pkg}.log"
        lg.addHandler(_make_rotating_handler(pkg_log, lvl, file_fmt))
        lg.setLevel(min(lvl, file_lvl, console_lvl))
        lg.propagate = False  # keep out of root.log

    # default: everyone else bubbles to root
    for name, lg in logging.Logger.manager.loggerDict.items():
        if not isinstance(lg, logging.Logger) or name in pkg_own_fs:
            continue
        lg.setLevel(logging.NOTSET)
        lg.propagate = True

    logging.captureWarnings(True)

    # ── 4) wire the queue, if present
    listener: Optional[logging.handlers.QueueListener] = None
    if early_queue is not None and early_handler is not None:
        # collect the final handlers currently on root (exclude the bootstrap QueueHandler)
        final_targets = [h for h in root.handlers if h is not early_handler]

        if queue_mode == "flush":
            # Drain the queue synchronously into the final handlers
            listener = logging.handlers.QueueListener(
                early_queue, *final_targets, respect_handler_level=True
            )
            listener.start()
            listener.stop()
            # bootstrap handler was already removed above
        else:  # "wire"
            # Steady-state queuing:
            # - leave early_handler as the ONLY root handler
            # - drive final_targets via a QueueListener
            # - to avoid double handling, remove final_targets from root
            for h in final_targets:
                root.removeHandler(h)
            listener = logging.handlers.QueueListener(
                early_queue, *final_targets, respect_handler_level=True
            )
            listener.start()

    # ── 5) announce AFTER queue is handled (so these log exactly once)
    if announce:
        _log_banner(root)  # one banner on console/root.log
        root.info("Central logging enabled → %s", root_log)
        for pkg, lvl in pkg_own_fs.items():
            root.info("  • %s → %s (level %s)", pkg, logs_dir / f"{pkg}.log", lvl)
        # Optional: per-package banners to files only
        for pkg in set(pkg_own_fs.keys()) | set(pkg_console_lvls.keys()):
            logging.getLogger(pkg).info(SESSION_START_BANNER, extra={"file_only": True})

    return listener
