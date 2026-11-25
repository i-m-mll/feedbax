"""Measure operands and Relate use the same places drawn by the Scene."""

import numpy as np
import pytest
from fastapi.testclient import TestClient

from feedbax.application.composition import load_application
from feedbax.contracts.expression_values import LENGTH, CallValue, LiteralValue, NameValue, ValueBase, ValueType
from feedbax.contracts.model_address import ModelAddress
from feedbax.contracts.model_document import ModelDocument
from feedbax.contracts.model_input import InitialModelSource, encode_document_input
from feedbax.contracts.objective_document import ObjectiveTerm
from feedbax.contracts.rollout import RolloutParameters
from feedbax.contracts.studio_trials import StudioTrialAuthoring
from feedbax.contracts.task_document import TaskDocument
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
    term_id = change["related_term"]
    saved = client.patch(url, json={"revision": body["revision"], "files": change["files"]})
    assert saved.is_success, saved.text
    after = client.get(url).json()
    assert after["project"]["documents"]["point-mass"] == snapshot["project"]["documents"]["point-mass"]
    terms = after["project"]["documents"]["simple-reaches-objective"]["payload"]["terms"]
    related_term = ObjectiveTerm.model_validate(next(term for term in terms if term["id"] == term_id))
    (difference,) = related_term.measure.args
    hand, target = difference.args
    prior = snapshot["project"]["documents"]["simple-reaches-objective"]["payload"]["terms"]
    existing = ObjectiveTerm.model_validate(next(term for term in prior if term["id"] == "effector_position"))
    assert hand == existing.measure.args[0].args[0]
    assert not ModelAddress.from_text(hand.reference).composites
    assert target.reference == "target"
    assert related_term.weight == LiteralValue(value=1)
    assert not related_term.epochs


def test_recorded_relation_uses_the_drawn_anchor_and_computed_point(project, tmp_path):
    client, url, body, snapshot, anchor = project
    shown = client.post(f"{url}/authoring/trials", json=body)
    assert shown.is_success, shown.text
    pairing = StudioTrialAuthoring.model_validate(shown.json())
    model = ModelDocument.model_validate(snapshot["project"]["documents"]["point-mass"]["payload"])
    assert isinstance(pairing.task, TaskDocument) and pairing.objective is not None
    hand = next(term.measure.args[0].args[0] for term in pairing.objective.terms if term.id == "effector_position")
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
        "task": pairing.task.model_dump(mode="json"),
        "objective": pairing.objective.model_copy(update={"terms": (*terms, lone)}).model_dump(mode="json"),
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
    objective = pairing.objective.model_copy(update={"terms": (*pairing.objective.terms, *terms)})
    store = ImmutableArtifactBlobProvider(tmp_path / "store")
    refs = {}
    for role, document in (
        ("subject", InitialModelSource(document=model)),
        ("task", pairing.task),
        ("objective", objective),
    ):
        refs[role], raw = encode_document_input(document)
        store.store_object(raw)
    trials = pairing.task.condition_ids[:1]
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
