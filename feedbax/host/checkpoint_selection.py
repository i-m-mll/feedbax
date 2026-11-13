"""Exact checkpoint publications and lineage bindings for the local Host."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from hashlib import sha256
import json
from pathlib import PurePosixPath

from feedbax.checkpoint_coordinates import cadence_coordinates, finite_coordinates
from feedbax.host.completion import CompletionFileError, verify_completion_files

EVALUATION_ENTRYPOINT = "feedbax.training.private_evaluation:main"


@dataclass(frozen=True)
class CheckpointSelection:
    coordinates: tuple[int, ...]
    requires_catalog: bool


def requested_checkpoint_selection(request: str, launch: dict) -> CheckpointSelection | None:
    """Read the coordinate demand from the canonical or private input binding."""
    payload = launch["payload"]
    if launch.get("warm_entrypoint") == EVALUATION_ENTRYPOINT:
        return CheckpointSelection(finite_coordinates(payload["coordinates"]), requires_catalog=True)
    if request.startswith("evaluate:"):
        binding = payload.get("canonical_request") or {}
        selector = (binding.get("checkpoint") or {}).get("select")
        if not isinstance(selector, dict):
            return None
        if selector.get("kind") == "batch":
            return CheckpointSelection(finite_coordinates((selector["batch"],)), requires_catalog=False)
        if selector.get("kind") == "batch_set":
            coordinates = finite_coordinates(selector["batches"])
            if (cadence := selector.get("cadence")) is not None and coordinates != cadence_coordinates(
                cadence["every_k"], cadence["through"],
            ):
                raise ValueError("checkpoint batches differ from the bounded cadence")
            return CheckpointSelection(coordinates, requires_catalog=True)
    return None


def selector_sha256(anchor: str, coordinates: Sequence[int]) -> str:
    return sha256(
        json.dumps([anchor, list(coordinates)], separators=(",", ":")).encode()
    ).hexdigest()


def verify_selected_publication(
    publication: dict,
    *,
    attempt: str,
    project: str,
    request: str,
    root: str,
    binding_sha256: str,
) -> dict:
    try:
        checkpoint = publication["checkpoint"]
        pointer = checkpoint["pointer"]
        update = checkpoint["completed_updates"]
        product = publication["checkpoint_product"]
        if (
            publication["key"] != f"checkpoint:{update}"
            or pointer["run_id"] != attempt
            or pointer["completed_training_batches"] != update
            or pointer["completed_coordinate"]["completed_barrier"] != "after_train_batch"
            or checkpoint["checkpoint_root"] != str(PurePosixPath(root) / "checkpoints" / attempt)
            or product["manifest_root"] != root
            or product["producer_job_id"] != attempt
            or product["project_id"] != project
            or product["request_id"] != request
            or product["binding_sha256"] != binding_sha256
            or product["checkpoint"] != checkpoint
        ):
            raise CompletionFileError(
                "Publication differs from the exact admitted checkpoint subject."
            )
        files = {item["path"]: item["sha256"] for item in product["files"]}
        relative = PurePosixPath("checkpoints") / attempt
        if (
            files.get(str(relative / "binding.json")) != binding_sha256
            or files.get(str(relative / pointer["manifest_relative_path"]))
            != pointer["manifest_sha256"]
        ):
            raise CompletionFileError("Publication lacks its exact binding or transaction bytes.")
        verify_completion_files(product)
        return product
    except (KeyError, TypeError, ValueError) as error:
        raise CompletionFileError(f"Invalid selected checkpoint publication: {error}") from error


def verify_dependency_files(binding: dict) -> None:
    try:
        if "checkpoint_selection" not in binding:
            verify_completion_files(binding["completion"])
            return
        selection = binding["checkpoint_selection"]
        requested = finite_coordinates(selection["requested"])
        lineage = selection["lineage"]
        if (
            selection["selector_sha256"] != selector_sha256(binding["dependency_job_id"], requested)
            or not lineage
            or lineage[0] != binding["dependency_job_id"]
            or lineage[-1] != selection["lineage_tip"]
            or len(lineage) != len(set(lineage))
            or len(selection["members"]) != len(requested)
        ):
            raise CompletionFileError("Malformed checkpoint selection binding.")
        for coordinate, product in zip(requested, selection["members"], strict=True):
            if (
                product["checkpoint"]["completed_updates"] != coordinate
                or product["producer_job_id"] not in lineage
                or type(product["publication_cursor"]) is not int
                or product["publication_cursor"] <= 0
            ):
                raise CompletionFileError(
                    "Selected checkpoint does not belong to its exact lineage."
                )
            verify_completion_files(product)
    except CompletionFileError:
        raise
    except (KeyError, TypeError, ValueError) as error:
        raise CompletionFileError(f"Cannot verify dependency binding: {error}") from error
