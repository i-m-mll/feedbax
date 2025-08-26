import argparse
import logging
import logging.handlers
import queue
from dataclasses import dataclass
from typing import Literal, Optional


@dataclass
class EarlyLogging:
    queue: queue.Queue[logging.LogRecord]
    handler: logging.handlers.QueueHandler


def setup_early_logging() -> EarlyLogging:
    """Install a QueueHandler on root if not present and return it."""
    root = logging.getLogger()
    # Reuse existing early handler if someone else already installed it.
    for h in root.handlers:
        if isinstance(h, logging.handlers.QueueHandler):
            return EarlyLogging(queue=h.queue, handler=h)  # type: ignore
    q: queue.Queue[logging.LogRecord] = queue.Queue(-1)
    h = logging.handlers.QueueHandler(q)
    root.setLevel(logging.DEBUG)  # capture everything; handlers will filter
    root.addHandler(h)
    return EarlyLogging(queue=q, handler=h)


def _early_parse(argv: list[str]) -> argparse.Namespace:
    """Parse early to determine running package."""
    p = argparse.ArgumentParser(add_help=False)
    p.add_argument("--single")
    p.add_argument("--batched")
    ns, _ = p.parse_known_args(argv)
    return ns


def setup(
    argv: list[str],
    *,
    domain: Literal["analysis", "training"],
    warn_dedup_default: bool = True,
    queue_mode: Literal["flush", "wire"] = "flush",
) -> Optional[str]:
    """Set up logging, warning de-dup, and globals config."""
    # Set up a queue handler to capture logs prior to full logging config.
    # Returns existing early logging if already initialized
    early = setup_early_logging()

    # Now that we're capturing logs, import other package components
    if warn_dedup_default:
        from feedbax_experiments._warnings import enable_warning_dedup

        enable_warning_dedup()

    from feedbax_experiments._logging import enable_logging_handlers
    from feedbax_experiments.config import configure_globals_for_package
    from feedbax_experiments.plugins import EXPERIMENT_REGISTRY

    # Infer package that will be run from CLI args, and load its globals
    args_ns = _early_parse(argv)
    if args_ns.single:
        pkg = EXPERIMENT_REGISTRY.resolve_package_for_module_key(args_ns.single, domain=domain)
    if args_ns.batched:
        pkg = EXPERIMENT_REGISTRY.resolve_package_for_batch_key(args_ns.batched, domain=domain)
    pkg = EXPERIMENT_REGISTRY.single_package_name()

    if pkg is not None:
        configure_globals_for_package(pkg, EXPERIMENT_REGISTRY)

    enable_logging_handlers(
        early_queue=early.queue,
        early_handler=early.handler,
        queue_mode=queue_mode,
    )
    return pkg
