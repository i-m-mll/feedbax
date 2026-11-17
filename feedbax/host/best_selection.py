"""Rank completed Train requests from their exact attested receipt bytes."""

from hashlib import sha256
from math import isfinite
from pathlib import Path

from feedbax.contracts.jobs import JobReceipt
from feedbax.contracts.manifest import canonical_manifest_path, training_run_manifest_id
from feedbax.contracts.strict_json import strict_json_loads
from feedbax.host.completion import CompletionFileError, completion_file


class BestMetricUnavailable(ValueError):
    """The selected measure is absent or not a finite number."""


def rank_train_metric(raw: bytes, receipt: JobReceipt, attempt: str, metric: str) -> float:
    """Read one finite measure from the immutable successful receipt of this attempt."""
    identity = training_run_manifest_id(attempt)
    if receipt.manifest_id != identity or sha256(raw).hexdigest() != receipt.manifest_sha256:
        raise CompletionFileError("Train receipt differs from exact completion evidence.")
    try:
        manifest = strict_json_loads(raw)
        if (
            manifest["kind"] != "TrainingRunManifest"
            or manifest["id"] != identity
            or manifest["job_id"] != attempt
            or manifest["status"] != "completed"
            or manifest.get("stopped", False)
        ):
            raise CompletionFileError(
                "Best ranking requires the producer's completed Train receipt."
            )
        metrics = manifest.get("summary_metrics", {})
        if not isinstance(metrics, dict):
            raise CompletionFileError("Train receipt summary_metrics must be an object.")
        score = metrics.get(metric)
    except CompletionFileError:
        raise
    except (ValueError, KeyError, TypeError) as error:
        raise CompletionFileError(f"Cannot measure exact Train receipt: {error}") from error
    if type(score) in (float, int):
        try:
            if isfinite(score):
                return float(score)
        except OverflowError:
            pass
    raise BestMetricUnavailable(f"Train attempt {attempt} has no finite {metric!r} metric.")


def completion_score(completion: dict, metric: str) -> float:
    try:
        root = Path(completion["manifest_root"])
        attempt = completion["producer_job_id"]
        receipt = JobReceipt.model_validate(completion["receipt"])
        path = canonical_manifest_path("TrainingRunManifest", receipt.manifest_id, root=root)
        relative = path.relative_to(root).as_posix()
        if not any(
            file["path"] == relative and file["sha256"] == receipt.manifest_sha256
            for file in completion["files"]
        ):
            raise CompletionFileError("Train receipt is absent from exact completion evidence.")
        with completion_file(root, relative) as (stream, _):
            raw = stream.read()
    except CompletionFileError:
        raise
    except (ValueError, OSError, KeyError, TypeError) as error:
        raise CompletionFileError(f"Cannot measure exact Train receipt: {error}") from error
    return rank_train_metric(raw, receipt, attempt, metric)
