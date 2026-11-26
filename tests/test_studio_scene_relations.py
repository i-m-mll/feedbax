"""Measure operands and Relate use the same places drawn by the Scene."""

import numpy as np
import pytest
from fastapi.testclient import TestClient

from feedbax.application.composition import load_application
from feedbax.contracts.expression_values import LENGTH, CallValue, LiteralValue, NameValue, ValueBase, ValueType
from feedbax.contracts.model_address import ModelAddress
from feedbax.contracts.stored_model import StoredModelDocument
from feedbax.contracts.model_input import InitialModelSource, encode_document_input
from feedbax.contracts.objective_document import ObjectiveTerm, StoredObjectiveTerm
from feedbax.contracts.rollout import RolloutParameters
from feedbax.contracts.studio_trials import StudioTrialAuthoring
from feedbax.contracts.task_document import StoredTaskDocument
from feedbax.host.evaluate_computation import EvaluationComputation, RolloutExecutables
from feedbax.host.evaluate_worker import _prepare_initial_rollout, _rollouts
from feedbax.persistence.artifact_custody import ImmutableArtifactBlobProvider
from feedbax.persistence.dataset import open_stored_dataset, publish_dataset
from feedbax.persistence.evaluation_trial_records import RecordedTrialLayout, read_trial_record
from feedbax.web.app import create_app


@pytest.fixture
def project(tmp_path):
    with TestClient(create_app(projects_dir=tmp_path / "projects")) as client:
        response = client.post("/api/projects", json={"name": "relation", "example": "point-mass-reaching"})
        assert response.status_code == 201, response.text
        snapshot = response.json()["snapshot"]
        url = f"/api/projects/{snapshot['project_id']}"
        body = {"revision": snapshot["revision"], "run": {
            "variant": "default", "step": "simple-reaches", "row": "default",
        }}
        geometry = client.post(f"{url}/scene/geometry", json={
            "revision": snapshot["revision"], "document_id": "point-mass", "poses": [{}],
        })
        assert geometry.is_success, geometry.text
        frame, = geometry.json()["frames"]
        anchor, = frame["anchors"]
        mechanism, = frame["mechanisms"]
        assert anchor["address"] == mechanism["point"]
        assert anchor["position"] == mechanism["position"]
        assert ModelAddress.model_validate(anchor["signal"]) == ModelAddress.at_state((), "plant", "effector/position")
        yield client, url, body, snapshot, anchor


def test_relate_uses_the_root_output_and_edits_only_the_objective(project):
    client, url, body, snapshot, anchor = project
    related = client.post(f"{url}/authoring/trials", json=body | {"change": {
        "kind": "relate", "first": {"kind": "anchor", "address": anchor["address"]},
        "second": {"kind": "variable", "id": "target"},
    }})
    assert related.is_success, related.text
    change = related.json()
    assert not change.get("store_choices")
    term_id = change["created_term"]
    saved = client.patch(url, json={"revision": body["revision"], "files": change["files"]})
    assert saved.is_success, saved.text
    after = client.get(url).json()
    assert after["project"]["documents"]["point-mass"] == snapshot["project"]["documents"]["point-mass"]
    terms = after["project"]["documents"]["simple-reaches-objective"]["payload"]["terms"]
    related_term = StoredObjectiveTerm.model_validate(terms[term_id])
    (difference,) = related_term.measure.args
    hand, target = difference.args
    prior = snapshot["project"]["documents"]["simple-reaches-objective"]["payload"]["terms"]
    existing = StoredObjectiveTerm.model_validate(prior["effector_position"])
    assert hand == existing.measure.args[0].args[0]
    assert hand.reference == "outputs.effector.pos"
    assert not ModelAddress.from_authored_text(hand.reference).composites
    assert target.reference == "target"
    assert related_term.weight == LiteralValue(value=1)
    assert related_term.window is None


def test_term_added_from_a_component_port_uses_its_root_output(project):
    client, url, body, _, _ = project
    field = {
        "kind": "field",
        "parent": {"kind": "port", "end": {
            "owner": "component", "component": "plant", "direction": "output", "port": "effector",
        }},
        "path": ["force"],
    }
    related = client.post(f"{url}/authoring/trials", json=body | {"change": {
        "kind": "relate", "first": {"kind": "port", "address": {"composites": [], "element": field}},
    }})
    assert related.is_success, related.text
    change = related.json()
    saved = client.patch(url, json={"revision": body["revision"], "files": change["files"]})
    assert saved.is_success, saved.text
    terms = saved.json()["project"]["documents"]["simple-reaches-objective"]["payload"]["terms"]
    term = StoredObjectiveTerm.model_validate(terms[change["created_term"]])
    root_port = {"kind": "port", "end": {"owner": "outputs", "port": "effector"}}
    root = ModelAddress.model_validate({"composites": [], "element": field | {"parent": root_port}})
    assert term.measure == CallValue(function="sqnorm", args=(NameValue(reference=root.to_authored_text()),))


def test_recorded_relation_uses_the_drawn_anchor_and_computed_point(project, tmp_path):
    client, url, body, snapshot, anchor = project
    shown = client.post(f"{url}/authoring/trials", json=body)
    assert shown.is_success, shown.text
    pairing = StudioTrialAuthoring.model_validate(shown.json())
    model = StoredModelDocument.model_validate(snapshot["project"]["documents"]["point-mass"]["payload"]).lower()
    assert isinstance(pairing.task, StoredTaskDocument) and pairing.objective is not None
    task = pairing.task.lower()
    source = pairing.objective.lower()
    hand = next(term for term in source.terms if term.id == "effector_position").measure.args[0].args[0]
    position = (0.14, -0.21)
    point = CallValue(function="point", args=tuple(LiteralValue(value=x, unit="m") for x in position))
    first, second = (NameValue(reference="target"), point)
    terms = [
        ObjectiveTerm(
            id=f"relative-{index}",
            measure=CallValue(function="sqnorm", args=(CallValue(function="sub", args=(hand, end)),)),
            measure_type=ValueType(base=ValueBase.REAL, dimensions=LENGTH.raised(2)),
        ) for index, end in enumerate((first, second))
    ]
    lone = ObjectiveTerm(
        id="fixed-point", measure=CallValue(function="sqnorm", args=(point,)),
        measure_type=ValueType(base=ValueBase.REAL, dimensions=LENGTH.raised(2)),
    )
    preview = client.post("/api/execution/task-trials/sample", json={
        "model": model.model_dump(mode="json"),
        "task": task.model_dump(mode="json"),
        "objective": source.model_copy(update={"terms": (*terms, lone)}).model_dump(mode="json"),
        "nominal": True,
    })
    assert preview.is_success, preview.text
    preview_terms = preview.json()["drawn"]["trials"][0]["terms"]
    assert preview_terms["relative-0"]["relations"] == [{
        "first": {"kind": "anchor", "address": anchor["address"]},
        "second": {"kind": "variable", "id": "target"},
    }]
    preview_place, = preview_terms["fixed-point"]["places"]
    assert preview_place["kind"] == "computed"
    assert preview_place["samples"][0] == pytest.approx(position)
    objective = source.model_copy(update={"terms": (*source.terms, *terms)})
    store = ImmutableArtifactBlobProvider(tmp_path / "store")
    refs = {}
    for role, document in (
        ("subject", InitialModelSource(document=model)),
        ("task", task),
        ("objective", objective),
    ):
        refs[role], raw = encode_document_input(document)
        store.store_object(raw)
    trials = task.condition_ids[:1]
    params = RolloutParameters(trials=trials, seed=12, trial_policy="paired", variation=())
    prepared = _prepare_initial_rollout(refs, params, registry=load_application().bundle, store=store)
    result = _rollouts(
        prepared, params, lambda: False, update_axis=None,
        computation=EvaluationComputation(store), executables=RolloutExecutables(),
    )
    assert result is not None
    saved = publish_dataset(store, result[0])
    with open_stored_dataset(store, saved, verify=True) as dataset:
        layout = RecordedTrialLayout.model_validate(dataset.descriptor.attrs["trial_record"])
        trial = read_trial_record(dataset, trials[0], (0, 0), layout)

    assert not trial.terms["effector_final_velocity"].relations
    reference, = trial.terms["relative-0"].relations
    assert reference.first.kind == "anchor" and reference.first.address.model_dump(mode="json") == anchor["address"]
    assert reference.second.kind == "variable" and reference.second.id == "target"
    relation, = trial.terms["relative-1"].relations
    assert relation.first == reference.first and relation.second.kind == "computed"
    assert np.asarray(relation.second.samples) == pytest.approx(
        np.broadcast_to(position, (len(trial.terms["relative-1"].steps), 2)),
    )
