import logging
import logging.handlers
import re
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Literal, Optional

import jax.tree as jt
from rich.highlighter import ReprHighlighter
from rich.logging import RichHandler
from rich.text import Text

from feedbax_experiments.config import LOGGING, PATHS
from feedbax_experiments.types import TreeNamespace

SESSION_START_BANNER = "―" * 20 + " NEW SESSION STARTED " + "―" * 20


def wire_queue(
    *,
    queue,
    bootstrap_handler: logging.Handler,
    mode: Literal["keep", "flush"] = "flush",
    respect_handler_level: bool = True,
) -> logging.handlers.QueueListener | None:
    """Connect a QueueHandler's queue to the current root handlers.
    - mode='flush': drain once and detach the bootstrap handler.
    - mode='keep' : start a QueueListener (steady-state) and keep queue-based logging.
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

    # keep
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
):
    # 0) unpack defaults, formatters, etc. (same as you have)
    file_lvl: int = file_level or LOGGING.file_level
    console_lvl: int = console_level or LOGGING.console_level
    pkg_console_lvls: dict[str, int] = pkg_console_levels or LOGGING.pkg_console_levels or {}
    pkg_own_fs: dict[str, int] = pkgs_own_files or LOGGING.pkgs_own_files or {}
    console_fmt = logging.Formatter(LOGGING.console_format_str)
    file_fmt = logging.Formatter(LOGGING.file_format_str)

    # 1) central dir + prune stray file handlers (same as you have)
    logs_dir = Path(PATHS.logs).resolve()
    logs_dir.mkdir(parents=True, exist_ok=True)
    _prune_foreign_file_handlers(logs_dir)

    root = logging.getLogger()
    root.setLevel(1)
    root_log = logs_dir / "root.log"

    def _log_session_start_banner(lg: logging.Logger, *, file_only: bool = False) -> None:
        payload = {"file_only": True} if file_only else {}
        if lg.level > logging.INFO:
            lg.log(lg.level, SESSION_START_BANNER, extra=payload)
        else:
            lg.info(SESSION_START_BANNER, extra=payload)

    # ---------- Phase A: queue-only logging ----------
    # Emit only ONE banner (root) while we still have only the QueueHandler
    _log_session_start_banner(root)
    # (Don't log per-package banners yet; some libs already have console handlers.)

    # ---------- Phase B: attach handlers ----------
    # Root file
    root.addHandler(_make_rotating_handler(root_log, file_lvl, file_fmt))

    # Root console (Rich) with "file_only" filter
    _remove_handlers(root, predicate=_console_handler_pred)
    console_h = RichHandler(level=console_lvl, highlighter=BacktickPathHighlighter())
    console_h.setFormatter(console_fmt)
    console_h.addFilter(_DropFileOnlyOnConsole())
    root.addHandler(console_h)

    # Per-package console overrides (each with the same filter), still propagate to root for files
    for pkg, lvl in (pkg_console_lvls or {}).items():
        lg = logging.getLogger(pkg)
        _remove_handlers(lg, predicate=_console_handler_pred)
        sh = RichHandler(level=lvl, highlighter=BacktickPathHighlighter())
        sh.setFormatter(console_fmt)
        sh.addFilter(_DropFileOnlyOnConsole())
        lg.addHandler(sh)
        lg.propagate = True

    # Per-package own files (no console; isolate)
    for pkg, lvl in (pkg_own_fs or {}).items():
        lg = logging.getLogger(pkg)
        _remove_handlers(lg, predicate=lambda h: isinstance(h, RotatingFileHandler))
        pkg_log = logs_dir / f"{pkg}.log"
        lg.addHandler(_make_rotating_handler(pkg_log, lvl, file_fmt))
        lg.setLevel(min(lvl, file_lvl, console_lvl))
        lg.propagate = False  # don't bubble to root

    # Default: everyone else bubbles to root
    for name, lg in logging.Logger.manager.loggerDict.items():
        if not isinstance(lg, logging.Logger) or name in (pkg_own_fs or {}):
            continue
        lg.setLevel(logging.NOTSET)
        lg.propagate = True

    # ---------- Announce per-pkg files (console once) ----------
    # These announcements are fine to the console (root logger), and go to root.log too.
    root.info("Central logging enabled → %s", root_log)
    for pkg, lvl in (pkg_own_fs or {}).items():
        root.info("  • %s → %s (level %s)", pkg, logs_dir / f"{pkg}.log", lvl)

    # ---------- Now log per-package banners, FILE-ONLY ----------
    for pkg in (pkg_own_fs or {}).keys() | (pkg_console_lvls or {}).keys():
        lg = logging.getLogger(pkg)
        _log_session_start_banner(lg, file_only=True)  # will be dropped by console filters

    # Capture warnings
    logging.captureWarnings(True)
