"""A synthesis method fills the shipped graph's declared model parameters."""

from dataclasses import asdict, replace
from base64 import b64decode

import jax
import numpy as np
from fastapi.testclient import TestClient

from feedbax.application.composition import load_application
from feedbax.builtins.synthesis import ConstantParametersSettings, constant_parameters
from feedbax.contracts.dataset import FieldDescriptor
from feedbax.contracts.model_address import ModelAddress
from feedbax.contracts.effective_success import success_content_ref
from feedbax.contracts.model_checkpoint import MODEL_TREE_MEDIA_TYPE, SemanticModelEncoding
from feedbax.contracts.model_origin import ModelProductKey, SynthesizedModelOrigin
from feedbax.contracts.operation import DatasetPort, JaxOperation
from feedbax.contracts.project import Project
from feedbax.contracts.request_identity import decode_request
from feedbax.contracts.request_specs import SynthesisRequestSpec
from feedbax.contracts.training import lower_stored_trial_documents
from feedbax.examples import load_shipped_projects
from feedbax.host.synthesis import PreparedSynthesis, synthesize_model
from feedbax.host import light_jobs, light_worker
from feedbax.jobs import JobService
from feedbax.persistence.array_container import decode_array_container
from feedbax.persistence.artifact_custody import ImmutableArtifactBlobProvider
from feedbax.persistence.effective_success import SuccessView, store_request
from feedbax.persistence.model_tree import MODEL_CHECKPOINT_STORE, read_model_tree
from feedbax.planner.project_plan import RunKey, compile_project
from feedbax.plugins.declarations import DistributionSource, callable_declaration
from feedbax.synthesis import synthesis_method
from feedbax.synthesis.methods import SynthesisMethodRegistry
from feedbax.dataset import Dataset, Variable
from feedbax.web.services.project_folder import ProjectFolderService
from feedbax.web.app import create_app
from tests.support.host import job_row, token, wait_until
from tests.support.managed_train_fixture import managed_evaluation_step
from tests.test_canonical_evaluate_admission import _save_project


def synthesis_project():
    raw = load_shipped_projects()["point-mass-reaching"].model_dump(mode="json", by_alias=True, exclude_unset=True)
    evaluate = managed_evaluation_step("synthesize")
    evaluate["inputs"]["subject"].pop("checkpoint")
    raw["steps"] = {
        "synthesize": {"label": "Synthesize", "kind": "synthesize", "base": {"inline": {
            "model": {"document": "point-mass"},
            "task": {"document": "simple-reaches-task"},
            "objective": {"document": "simple-reaches-objective"},
            "method": "feedbax.constant_parameters",
        }}},
        "baseline": evaluate,
    }
    return raw


def test_synthesize_and_evaluate_plan_from_authored_documents():
    raw = synthesis_project()
    plan = compile_project(Project.model_validate(raw))
    assert not plan.diagnostics, plan.diagnostics
    runs = {run.kind: run for run in plan.runs}
    assert set(runs) == {"synthesize", "evaluate"}
    assert not any(run.admission_problem for run in runs.values())
    synthesis = runs["synthesize"]
    decoded = decode_request(plan.requests[synthesis.request_id], plan.documents.__getitem__)
    assert isinstance(decoded.spec, SynthesisRequestSpec)
    assert decoded.spec.settings == {"value": 0}
    chosen = runs["evaluate"].inputs["subject"].member.reference
    assert chosen.request_id == synthesis.request_id and chosen.role == "model" and chosen.select is None

    raw["documents"]["point-mass"]["payload"]["interior"]["components"]["plant"]["fields"]["mass"] = 1.5
    changed = compile_project(Project.model_validate(raw))
    before = {run.key: run.request_id for run in plan.runs}
    assert len(changed.runs) == len(before)
    assert all(run.request_id != before[run.key] for run in changed.runs)
    raw["steps"]["synthesize"]["base"]["inline"]["input_bindings"] = {"target_pos": None}
    refused = compile_project(Project.model_validate(raw)).row(synthesis.key)
    assert any("needs a Task signal" in problem.detail for problem in refused.problems)


def test_synthesize_resolves_model_sweeps_and_trial_set_fields():
    raw = synthesis_project()
    raw["steps"].pop("baseline")
    mass = ModelAddress.at_stored_path(("interior", "components", "plant", "fields", "mass"))
    step = raw["steps"]["synthesize"]
    step["sweep"] = {"axes": {"mass": {"values": {
        name: {"value": value, "patches": {mass.to_text(): {"value": value}}}
        for name, value in (("light", 1.0), ("heavy", 1.5))
    }}}}
    raw["documents"]["bank"] = {
        "label": "Bank", "schema": "feedbax.set_document.v2",
        "payload": {"entries": {"one": {"label": "One"}, "two": {"label": "Two"}}},
    }
    step["set_fields"] = {"bank": {"axis": {"set": "bank"}, "sampling": {"kind": "weighted", "none": 0}}}
    plan = compile_project(Project.model_validate(raw))
    assert not plan.diagnostics, plan.diagnostics
    assert len(plan.runs) == 2 and not any(run.admission_problem for run in plan.runs)
    for run in plan.runs:
        spec = decode_request(plan.requests[run.request_id], plan.documents.__getitem__).spec
        assert isinstance(spec, SynthesisRequestSpec)
        documents = spec.trial_documents
        assert documents.model.interior.components["plant"].fields["mass"] == run.coordinates["mass"].value
        assert any(variable.set_choices is not None for variable in documents.task.variables)


def test_served_synthesis_documents_bind_in_trials_and_model_view(host, tmp_path, monkeypatch):
    monkeypatch.setenv("FEEDBAX_HOST_ROOT", str(host.root))
    raw = synthesis_project()
    project = "synthesis-authoring"
    raw["documents"]["extra-model"] = raw["documents"]["point-mass"]
    signals = raw["documents"]["simple-reaches-task"]["payload"]["signals"]
    signals["alternate_target"] = signals["target_pos"] | {"label": "Alternate target"}
    _save_project(ProjectFolderService(tmp_path), project, Project.model_validate(raw))
    route = f"/api/projects/{project}"
    run = {"variant": "default", "step": "synthesize", "row": "default"}
    with TestClient(create_app(projects_dir=tmp_path, automatic_run="off")) as client:
        revision = client.get(route).json()["revision"]
        body = {"revision": revision, "run": run}
        response = client.post(f"{route}/authoring/trials", json=body)
        assert response.status_code == 200, response.text
        view = response.json()
        assert view["model"] == "point-mass" and view["task_document"] == "simple-reaches-task"
        assert view["objective_document"] == "simple-reaches-objective" and not view["problems"]
        target = next(port for port in view["inputs"] if port["id"] == "target_pos")
        assert (target["signal_id"], target["explicit"]) == ("target_pos", False)
        assert view["epoch_instances"]
        response = client.post(f"{route}/model/projection", json=body | {"document": "point-mass"})
        assert response.status_code == 200, response.text
        assert response.json()["task_signals"]
        response = client.get(f"{route}/train-runs", params={"revision": revision, "window_map": True})
        assert response.status_code == 200, response.text
        window = response.json()["window_map"]
        selected = next(row for row in window["rows"] if row["key"] == run)
        assert selected["evaluation_conditions"]
        step = next(step for step in window["steps"] if step["step_id"] == run["step"])
        assert step["document_ids"] == {
            "payload.model": "point-mass", "payload.task": "simple-reaches-task",
            "payload.objective": "simple-reaches-objective",
        }
        for role, document in (
            ("model", "extra-model"), ("task", "delayed-reaches-task"), ("objective", "delayed-reaches-objective"),
        ):
            response = client.post(f"{route}/authoring/trials", json=body | {
                "change": {"kind": "select_document", "role": role, "document": document},
            })
            assert response.status_code == 200, response.text
            files = response.json()["files"]
            assert len(files) == 1 and files[0]["address"] == {"kind": "step", "id": "synthesize"}, response.json()
            assert files[0]["patches"] == [{
                "op": "replace", "path": "/base/inline",
                "value": raw["steps"]["synthesize"]["base"]["inline"] | {role: {"document": document}},
            }]
        binding = client.post(f"{route}/authoring/trials", json=body | {
            "change": {"kind": "binding", "input_id": "target_pos", "signal_id": "alternate_target"},
        })
        assert binding.status_code == 200, binding.text
        assert binding.json()["input_bindings"] == {"target_pos": "alternate_target"}
        assert binding.json()["files"][0]["patches"][0]["value"]["input_bindings"] == {"target_pos": "alternate_target"}


def test_saved_synthesize_model_runs_evaluation_train_start_and_export(host, tmp_path, monkeypatch):
    registry = load_application().bundle
    jobs = JobService(host, registry)
    folders = ProjectFolderService(tmp_path / "projects")
    project = "synthesized-reaching"
    raw = synthesis_project()
    shipped = load_shipped_projects()["point-mass-reaching"]
    train = shipped.steps["simple-reaches"].model_dump(mode="json", by_alias=True, exclude_unset=True)
    raw["steps"]["refine"] = train
    raw["steps"]["synthesize"]["base"]["inline"]["settings"] = {"value": 0.0}
    raw["steps"]["synthesize"]["sweep"] = {"axes": {"fill": {"values": {
        name: {"value": value, "patches": {"settings.value": {"value": value}}}
        for name, value in (("zero", 0.0), ("offset", 0.125))
    }}}}
    source = {"step": "synthesize", "where": {"clauses": [{"fill": [{"id": "zero", "label": "Zero"}]}]}}
    raw["steps"]["baseline"]["inputs"]["subject"] = source
    _save_project(folders, project, Project.model_validate(raw))
    monkeypatch.setenv("FEEDBAX_HOST_ROOT", str(host.root))
    route = f"/api/projects/{project}"
    with TestClient(create_app(projects_dir=folders.root, automatic_run="off")) as client:
        revision = client.get(route).json()["revision"]
        start_url = f"{route}/authoring/train/refine/start"
        response = client.post(start_url, json={"revision": revision, "source": source})
        assert response.status_code == 200, response.text
        edits = response.json()
        assert edits["files"] and not edits["problems"]
        invalid = client.post(start_url, json={"revision": revision, "source": source | {"checkpoint": "last"}})
        assert invalid.status_code == 422, invalid.text
        saved = client.patch(route, json={"revision": revision, "files": edits["files"]})
        assert saved.status_code == 200, saved.text
    plan = folders.read_plan(project, revision=None)
    synthesis = RunKey("default", "synthesize", "fill=zero")
    refine = RunKey("default", "refine", "default")
    assert plan.plan.row(refine).inputs["checkpoint"].producer == synthesis
    assert folders.read(project).project.steps["refine"].base.inline["checkpoint_initialization"] == "initialize_from"
    (admitted,) = jobs.admit_plan(folders, plan, run_keys=(synthesis,), token=token("synthesize")).items
    assert admitted.disposition == "admitted", admitted
    job = wait_until(lambda: (
        job if (job := job_row(host, admitted.job_id)) and job["outcome"] is not None else None
    ))
    assert job["state"] == "succeeded", job["outcome"]
    store = ImmutableArtifactBlobProvider(plan.store_path)
    view = SuccessView(store)
    receipt = view.read_attempt(admitted.job_id).receipt
    assert tuple(receipt.products) == ("model",)
    product, = receipt.products["model"]
    assert product.content.tree and product.content.media_type == MODEL_TREE_MEDIA_TYPE
    assert receipt.request.id.startswith("synthesize:")
    model = read_model_tree(store, success_content_ref(product.content), SemanticModelEncoding.model_validate(product.descriptor))
    replica, = model.replicas
    parameters = decode_array_container(
        store.get_object(replica.arrays), expected_descriptor_sha256=replica.descriptor.array_descriptor_sha256,
    ).tensors
    assert parameters and all(value.dtype == np.float64 for value in parameters.values())

    evaluate = RunKey("default", "baseline", "default")
    (rolled_out,) = jobs.admit_plan(folders, plan, run_keys=(evaluate,), token=token("evaluate-synthesis")).items
    assert rolled_out.disposition == "admitted", rolled_out
    job = wait_until(lambda: (
        job if (job := job_row(host, rolled_out.job_id)) and job["outcome"] is not None else None
    ))
    assert job["state"] == "succeeded", job["outcome"]
    result = jobs.read_evaluation_result(project=plan.project_id, store=plan.store_path, job_id=rolled_out.job_id)
    assert result._manifest().model_origin == SynthesizedModelOrigin(
        kind="synthesized", key=ModelProductKey(variant="default", step="synthesize", row=synthesis.row),
        job_id=admitted.job_id, request_id=receipt.request.id,
    )
    with result.open_states() as dataset:
        assert "checkpoint" not in dataset.descriptor.dimensions
        assert dataset.descriptor.dimensions["trial"] > 0

    (trained,) = jobs.admit_plan(folders, plan, run_keys=(refine,), token=token("train-synthesis")).items
    assert trained.disposition == "admitted", trained
    job = wait_until(lambda: (job if (job := job_row(host, trained.job_id)) and job["outcome"] is not None else None))
    assert job["state"] == "succeeded", job["outcome"]
    (start,) = SuccessView(store).read_attempt(trained.job_id).receipt.inputs["checkpoint"]
    assert start.content == product.content and start.selection.receipt == view.attempt_ref(admitted.job_id)

    with TestClient(create_app(projects_dir=folders.root, automatic_run="off")) as client:
        export_url = f"{route}/loaded-models/export"
        body = {"revision": plan.revision, "key": {"variant": "default", "step": "synthesize", "row": synthesis.row},
                "job_id": admitted.job_id}
        archive = client.post(export_url, json=body)
        assert archive.status_code == 200, archive.text
        assert archive.content[:2] == b"PK"
        assert archive.headers["content-disposition"] == 'attachment; filename="synthesize.feedbax-model"'
        assert client.post(export_url, json=body | {"completed_updates": 2}).status_code == 422


def test_synthesis_writes_one_replica_with_every_parameter(tmp_path):
    project = load_shipped_projects()["point-mass-reaching"]
    spec = SynthesisRequestSpec(
        trial_documents=lower_stored_trial_documents({
            "model": project.documents["point-mass"].payload,
            "task": project.documents["simple-reaches-task"].payload,
            "objective": project.documents["simple-reaches-objective"].payload,
            "input_bindings": {"target_pos": "target_pos"},
        }),
        method="feedbax.constant_parameters",
        settings={"value": 0.125},
    )
    registry = load_application().bundle
    artifacts = ImmutableArtifactBlobProvider(tmp_path / "artifacts")
    transaction = tmp_path / "attempt"
    with jax.enable_x64():
        result = synthesize_model(spec, registry=registry, artifacts=artifacts)
        assert isinstance(result, PreparedSynthesis), result
        published = result.writer.publish(transaction, (result.model,), templates=(result.template,), replica_ids=("0",))
    store = ImmutableArtifactBlobProvider(transaction / MODEL_CHECKPOINT_STORE)
    model = read_model_tree(store, published.tree, published.encoding)
    replica, = model.replicas
    assert replica.descriptor.id == "0"
    arrays = decode_array_container(
        store.get_object(replica.arrays), expected_descriptor_sha256=replica.descriptor.array_descriptor_sha256,
    )
    parameters = [
        arrays.tensors[address] for address, metadata in arrays.descriptor.metadata.items()
        if metadata["representation"] == "jax-array"
    ]
    assert parameters
    for array in parameters:
        assert array.dtype == np.float64
        np.testing.assert_equal(array, np.full_like(array, 0.125))

    restricted = spec.model_copy(update={"target": ModelAddress.at_stored_path(("interior", "components", "plant"))})
    with jax.enable_x64():
        problems = synthesize_model(restricted, registry=registry, artifacts=artifacts)
    assert isinstance(problems, tuple) and problems
    assert all(problem.model_address is not None and "parameters" in problem.path for problem in problems)


def test_analyze_reads_a_declared_synthesis_dataset(tmp_path, monkeypatch):
    schema = "feedbax.synthesis.test.v1"
    port = DatasetPort(
        schema_ids=(schema,), coordinates={"sample": FieldDescriptor(dims=("sample",))},
        variables={"value": FieldDescriptor(dims=("sample",))},
    )

    @synthesis_method(
        settings=ConstantParametersSettings, numerical=JaxOperation(backend="cpu", floating_point_bits=64),
        outputs={"diagnostic": port},
    )
    def synthesize_with_diagnostic(problem, settings):
        dataset = Dataset.from_arrays(
            {"value": Variable(np.array([1., 3.]), ("sample",))},
            coords={"sample": Variable(np.array([0, 1]), ("sample",))}, schema_id=schema,
        )
        result = constant_parameters(problem, settings)
        return replace(result, datasets={"diagnostic": dataset})

    methods = SynthesisMethodRegistry()
    key = "feedbax.test_diagnostic"
    methods.register_declaration(
        synthesize_with_diagnostic, callable_declaration(synthesize_with_diagnostic),
        key=key, source=DistributionSource("feedbax", "feedbax", "test"),
    )
    methods.seal()
    registry = replace(load_application().bundle, synthesis_methods=methods)
    monkeypatch.setattr(light_worker, "operation_application", lambda: registry)

    raw = synthesis_project()
    raw["steps"]["synthesize"]["base"]["inline"]["method"] = key
    raw["steps"]["plot"] = {
        "label": "Plot",
        "kind": "analyze", "inputs": {"diagnostic": {"step": "synthesize", "product": "diagnostic"}},
        "base": {"inline": {
            "inputs": {"diagnostic": {"kind": "dataset"}},
            "views": {"result": {"label": "Result", "inputs": {"data": {"input": "diagnostic"}}, "figure": {
                "data": ["data"], "layout": {"values": {"row": 1, "col": 1}},
                "layers": {"values": {"role": "data", "panel": "values", "mark": "line",
                                      "encode": {"x": "sample", "y": "value"}}},
            }}},
        }},
    }
    plan = compile_project(Project.model_validate(raw), operations=registry.operations, synthesis_methods=methods)
    assert not plan.diagnostics, plan.diagnostics
    element, = plan.elements.values()
    (binding,) = element.inputs["data"]
    source_id = binding.member.reference.request_id
    assert binding.member.reference.role == "diagnostic"

    store = ImmutableArtifactBlobProvider(tmp_path / "store")

    def request(id_):
        decoded = decode_request(plan.requests[id_], plan.documents.__getitem__)
        from feedbax.contracts.request_identity import encode_request

        return decoded, store_request(store, encode_request(decoded.spec, decoded.inputs))

    source, source_object = request(source_id)
    role, provider = light_jobs.validate_light_request(source, registry=light_worker.operation_application())
    with jax.enable_x64():
        products = light_worker._compute_products(store, source, {}, "synthesized", role, provider)
    published = light_jobs.publish_light_success(
        light_jobs.LightDelivery(
            project_id="test", manifest_root=store.root, request=source_object,
            declaration=light_jobs.AdmittedDeclaration.of(provider),
        ).model_dump(mode="json"),
        "synthesized", source_id, {}, products, provider=asdict(provider.source),
    )
    assert set(products) == {"model", "diagnostic"}

    render, render_object = request(element.request_id)
    from feedbax.contracts.artifact_custody import ContentObjectRef
    from feedbax.contracts.strict_json import strict_json_loads

    _, inputs = SuccessView(store).bind(render_object, origins={source_id: ContentObjectRef.model_validate(published)})
    output = strict_json_loads(light_worker.render_saved_figure(store, render, inputs))
    y = output["data"][0]["y"]
    np.testing.assert_equal(np.frombuffer(b64decode(y["bdata"]), dtype=y["dtype"]), [1.0, 3.0])
