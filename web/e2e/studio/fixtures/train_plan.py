"""Read the acknowledged static Plan and round-trip its Train request."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from feedbax.contracts.project_plan import RunKey
from feedbax.contracts.request_identity import decode_request, encode_request
from feedbax.contracts.request_specs import TrainingRequestSpec
from feedbax.contracts.studio_training import StudioGraphMethodPayload
from feedbax.plugins.application import new_application_registry_bundle
from feedbax.web.services.project_folder import ProjectFolderService


def inspect(root: Path, revision: str) -> dict:
    service = ProjectFolderService(root.parent)
    key = RunKey("default", "train", "default")
    settled = service.read_plan(root.name, revision=revision, source_runs=[key])
    plan = settled.plan
    result = {
        "revision": settled.revision,
        "semanticRoot": settled.semantic_root,
        "storeExists": settled.store_path.exists(),
        "rows": len(plan.rows),
        "runs": len(plan.runs),
        "diagnostics": sorted([problem.code, problem.path] for problem in plan.diagnostics),
    }
    if not plan.runs:
        return result

    (run,) = plan.runs
    assert run.key == key
    request_bytes = plan.requests[run.request_id]
    decoded = decode_request(request_bytes, plan.documents.__getitem__)
    spec = decoded.spec
    assert isinstance(spec, TrainingRequestSpec)
    assert decoded.id == run.request_id
    assert not decoded.inputs
    encoded = encode_request(spec, decoded.inputs)
    assert (encoded.id, encoded.data) == (run.request_id, request_bytes)
    loss = spec.objective.loss
    assert loss is not None and loss.time_agg is not None
    assert spec.task_binding_spec is not None
    method = new_application_registry_bundle().training_programs.validate_payload(
        spec.method_ref,
        spec.method_payload,
        path="/method_payload",
    )
    assert isinstance(method, StudioGraphMethodPayload)
    assert method.target_total == spec.training_config.n_batches
    assert method.optimizer.params["learning_rate"] == spec.training_config.learning_rate
    sources = plan.sources[key]
    return {
        **result,
        "requestId": decoded.id,
        "model": spec.graph.nodes["plant"].type,
        "task": spec.task.model_dump(mode="json", exclude_none=True),
        "binding": spec.task_binding_spec.bindings[0].source_data_id,
        "objective": [
            loss.type,
            loss.label,
            loss.selector,
            loss.target_selector,
            loss.time_agg.mode,
        ],
        "method": spec.method_ref.key,
        "optimizer": method.optimizer.type,
        "seed": method.seed,
        "config": spec.training_config.model_dump(
            mode="json",
            include={
                "n_batches",
                "batch_size",
                "learning_rate",
                "n_reach_steps",
                "snapshot_interval",
            },
        ),
        "sources": [
            [sources[path].kind, sources[path].name, sources[path].path]
            for path in (
                "/payload/graph/nodes/plant/type",
                "/payload/task/type",
                "/payload/objective/loss/selector",
                "/payload/method_ref/name",
                "/payload/method_payload/payload/target_total",
                "/payload/training_config/n_batches",
            )
        ],
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("revision")
    args = parser.parse_args()
    print(json.dumps(inspect(args.root, args.revision)))
