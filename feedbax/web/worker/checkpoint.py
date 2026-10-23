"""Internal custody helpers for worker-owned checkpoint directories."""

from __future__ import annotations

import os
import shutil


class CheckpointCleanupError(RuntimeError):
    """Report a worker checkpoint directory that could not be removed."""

    def __init__(
        self,
        checkpoint_path: str,
        *,
        context: str,
        cleanup_error: Exception,
    ) -> None:
        self.checkpoint_path = checkpoint_path
        self.checkpoint_directory = os.path.dirname(checkpoint_path)
        self.context = context
        self.cleanup_error = cleanup_error
        super().__init__(
            f"{context}: could not remove worker checkpoint directory "
            f"{self.checkpoint_directory!r}; residual checkpoint path "
            f"{checkpoint_path!r} remains available for cleanup retry; "
            f"cleanup error was {type(cleanup_error).__name__}: {cleanup_error}"
        )


def cleanup_worker_checkpoint(checkpoint_path: str | None, *, context: str) -> bool:
    """Remove one worker-owned checkpoint directory or report its residual path.

    Returns whether *checkpoint_path* named a managed worker checkpoint. Missing
    managed directories count as already cleaned. Cleanup failures raise with
    the residual path so the caller can retain or restore registry custody.
    """
    if checkpoint_path is None:
        return False
    checkpoint_dir = os.path.dirname(checkpoint_path)
    if not os.path.basename(checkpoint_dir).startswith("feedbax_ckpt_"):
        refusal = ValueError("checkpoint path is not inside a worker-managed directory")
        raise CheckpointCleanupError(
            checkpoint_path,
            context=context,
            cleanup_error=refusal,
        ) from refusal
    try:
        shutil.rmtree(checkpoint_dir)
    except FileNotFoundError:
        return True
    except OSError as exc:
        raise CheckpointCleanupError(
            checkpoint_path,
            context=context,
            cleanup_error=exc,
        ) from exc
    return True
