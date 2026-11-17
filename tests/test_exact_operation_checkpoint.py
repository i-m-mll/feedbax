"""An Analyze model role uses its chosen publication after Train has finished."""

import os
import shutil

from feedbax.contracts.artifact_custody import ContentObjectRef
from feedbax.contracts.project import Project
from feedbax.contracts.project_io import ProjectConfiguration, dump_text_project
from feedbax.jobs import JobService
from feedbax.light_jobs import dataset_member
from feedbax.persistence.artifact_custody import ImmutableArtifactBlobProvider
from feedbax.persistence.dataset import open_stored_dataset
from feedbax.persistence.effective_success import SuccessView
from feedbax.plugins.composition import load_application
from feedbax.project_folder import ProjectFolder
from tests.support.managed_train_fixture import managed_train_project
from tests.test_local_job_host import host, job_row, wait_until  # noqa: F401 -- pytest fixture


def test_exact_model_operation_uses_earlier_publication_and_refuses_missing(request, tmp_path, monkeypatch):
    package = tmp_path / "checkpoint_probe"
    package.mkdir()
    (package / "__init__.py").write_text('''
from typing import Annotated

import equinox as eqx
import jax.numpy as jnp
from pydantic import BaseModel

from feedbax.analysis.operations import DatasetOutput, operation
from feedbax.contracts.operation import DatasetPort, JaxOperation, ModelPort
from feedbax.dataset import Dataset, Variable
from feedbax.runtime.graph import Graph
from feedbax.training.methods import TrainingState, training_method


class Settings(BaseModel):
    pass


def weight(model):
    return model.nodes["readout"].layer.weight


def initialize(settings, model, key):
    return TrainingState(eqx.tree_at(weight, model, replace_fn=jnp.ones_like), ())


@training_method(settings=Settings, init=initialize)
def increment(state, ctx):
    return TrainingState(eqx.tree_at(weight, state.model, replace_fn=lambda value: value + 1), ()), {}


@operation(settings=Settings, numerical=JaxOperation(backend="cpu", floating_point_bits=32))
def read_weight(model: Annotated[Graph, ModelPort()], params: Settings) -> Annotated[
    Dataset, DatasetOutput("weight", DatasetPort(schema_ids=("checkpoint_probe.weight.v1",))),
]:
    return Dataset.from_arrays(
        {"weight": Variable(weight(model)[0, 0], ())}, schema_id="checkpoint_probe.weight.v1",
    )
''')
    metadata = tmp_path / "checkpoint_probe-1.0.dist-info"
    metadata.mkdir()
    (metadata / "METADATA").write_text("Name: checkpoint-probe\nVersion: 1.0\n")
    (metadata / "entry_points.txt").write_text("[feedbax.plugins]\nprobe = checkpoint_probe\n")
    monkeypatch.syspath_prepend(str(tmp_path))
    monkeypatch.setenv("PYTHONPATH", os.pathsep.join(filter(None, (str(tmp_path), os.environ.get("PYTHONPATH")))))
    monkeypatch.setenv("JAX_PLATFORMS", "cpu")

    host = request.getfixturevalue("host")
    host.policy(compute=1)
    registries = load_application().bundle
    folders = ProjectFolder(
        tmp_path / "projects", components=registries.components, python_tasks=registries.python_tasks,
        operations=registries.operations, functions=registries.expression_functions,
        training_methods=registries.training_methods,
    )
    jobs = JobService(host, registries)
    project = "checkpoint-selection"
    raw = managed_train_project(batches=3).model_dump(mode="json", by_alias=True, exclude_unset=True)
    authored = raw["steps"]["train"]["base"]["inline"]
    authored.update(method="checkpoint_probe.increment", settings={}, seed=7)
    authored["training_config"].update(n_batches=3, batch_size=1, snapshot_interval=2, replicates=1)

    def save():
        for name, data in dump_text_project(
            Project.model_validate(raw), ProjectConfiguration(project=project, store="results"),
        ).items():
            path = folders.root / project / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(data)
        review = folders.read_plan(project, revision=None)
        assert not review.plan.diagnostics, review.plan.diagnostics
        return review

    def run(review, *steps):
        keys = tuple(row.key for row in review.plan.rows if row.key.step in steps)
        result = jobs.admit_plan(folders, review, run_keys=keys, token="-".join(steps))
        assert len(result.items) == len(steps)
        assert all(item.job_id is not None and item.disposition == "admitted" for item in result.items), result

        def settled():
            rows = [job_row(host, item.job_id) for item in result.items]
            return rows if all(row is not None and row["outcome"] is not None for row in rows) else None

        return wait_until(settled, timeout=35)

    (trained,) = run(save(), "train")
    assert trained["state"] == "succeeded", trained["outcome"]
    raw["documents"]["read-model"] = {"schema": "feedbax.analysis_document.v1", "payload": {
        "inputs": {"model": {"kind": "model"}},
        "operations": {"read": {
            "operation": "checkpoint_probe.read_weight", "inputs": {"model": {"input": "model"}},
        }},
    }}
    for name, selector in (("earlier", {"at": 2}), ("last", "last"), ("missing", {"at": 1})):
        raw["steps"][name] = {
            "kind": "analyze", "base": {"document": "read-model"},
            "inputs": {"model": {"step": "train", "checkpoint": selector}},
        }
    review = save()
    (earlier,) = run(review, "earlier")
    (last,) = run(review, "last")
    (missing,) = run(review, "missing")
    assert earlier["state"] == last["state"] == "succeeded", (earlier["outcome"], last["outcome"])
    assert missing["state"] == "blocked" and missing["outcome"]["error"]["code"] == "input_unavailable"

    store = ImmutableArtifactBlobProvider(review.store_path)
    shutil.rmtree(store.root / "checkpoints" / trained["id"])
    for row, expected in ((earlier, 3), (last, 4)):
        ref = ContentObjectRef.model_validate(row["outcome"]["store_success"])
        (product,) = SuccessView(store).read_success(ref).receipt.products["weight"]
        with open_stored_dataset(store, dataset_member(product)) as data:
            assert data.data("weight").item() == expected
