"""Exact checkpoint publications and lineage bindings for the local Host."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from hashlib import sha256
import json
from pathlib import PurePosixPath

from feedbax.checkpoint_coordinates import cadence_coordinates, finite_coordinates
from feedbax.host.completion import CompletionFileError, verify_completion_files


@dataclass(frozen=True)
class CheckpointSelection:
    coordinates: tuple[int, ...]
    requires_catalog: bool
    allows_gaps: bool = False


def is_checkpoint_producer(launch: dict, *, continues: str | None, manifest_root: str) -> bool:
    payload = launch["payload"]
    return (
        continues is None
        and payload.get("canonical_request") is not None
        and payload.get("manifest_root") == manifest_root
    )


def requested_checkpoint_selection(request: str, launch: dict) -> CheckpointSelection | None:
    """Read the coordinate demand from the canonical input binding."""
    payload = launch["payload"]
    if request.startswith("evaluate:"):
        binding = payload.get("canonical_request") or {}
        selector = (binding.get("checkpoint") or {}).get("select")
        if not isinstance(selector, dict):
            return None
        if selector.get("kind") == "batch":
            return CheckpointSelection(finite_coordinates((selector["batch"],)), requires_catalog=False)
        if selector.get("kind") == "update_series":
            return CheckpointSelection(finite_coordinates(selector["updates"]), False, allows_gaps=True)
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
        verify_completion_files(product, checkpoint_model=True)
        return product
    except (KeyError, TypeError, ValueError) as error:
        raise CompletionFileError(f"Invalid selected checkpoint publication: {error}") from error


def selected_coordinates(selection: dict) -> tuple[int, ...]:
    requested = finite_coordinates(selection["requested"])
    gaps = selection.get("gaps", ())
    missing = {gap["update"] for gap in gaps}
    if (
        len(missing) != len(gaps)
        or not missing <= set(requested)
        or any(type(gap["update"]) is not int or gap["reason"] != "checkpoint_unavailable" for gap in gaps)
        or [gap["update"] for gap in gaps] != [update for update in requested if update in missing]
    ):
        raise CompletionFileError("Malformed checkpoint selection gaps.")
    return tuple(update for update in requested if update not in missing)


def verify_dependency_files(binding: dict) -> None:
    try:
        if "checkpoint_selection" not in binding:
            completion = binding["completion"]
            verify_completion_files(completion, checkpoint_model=completion["request_id"].startswith("train:"))
            return
        selection = binding["checkpoint_selection"]
        requested = finite_coordinates(selection["requested"])
        available = selected_coordinates(selection)
        lineage = selection["lineage"]
        if (
            selection["selector_sha256"] != selector_sha256(binding["dependency_job_id"], requested)
            or not lineage
            or lineage[0] != binding["dependency_job_id"]
            or lineage[-1] != selection["lineage_tip"]
            or len(lineage) != len(set(lineage))
            or len(selection["members"]) != len(available)
        ):
            raise CompletionFileError("Malformed checkpoint selection binding.")
        if "gaps" in selection:
            closure = selection["closure"]
            product = closure["product"]
            if (
                closure["kind"] not in ("completion", "recovery")
                or product["producer_job_id"] != selection["lineage_tip"]
                or (closure["kind"] == "completion" and not product.get("receipt"))
                or (closure["kind"] == "recovery" and any(
                    gap["update"] > product["checkpoint"]["completed_updates"]
                    for gap in selection["gaps"]
                ))
            ):
                raise CompletionFileError("Malformed checkpoint coverage closure.")
            verify_completion_files(product)
        for coordinate, product in zip(available, selection["members"], strict=True):
            if (
                product["checkpoint"]["completed_updates"] != coordinate
                or product["producer_job_id"] not in lineage
                or type(product["publication_cursor"]) is not int
                or product["publication_cursor"] <= 0
            ):
                raise CompletionFileError(
                    "Selected checkpoint does not belong to its exact lineage."
                )
            verify_completion_files(product, checkpoint_model=True)
    except CompletionFileError:
        raise
    except (KeyError, TypeError, ValueError) as error:
        raise CompletionFileError(f"Cannot verify dependency binding: {error}") from error
