"""Observable computation and driver evidence for managed rollout scoring."""

from contextlib import ExitStack
from dataclasses import replace
from datetime import UTC, datetime
from hashlib import sha256
from importlib.util import module_from_spec, spec_from_file_location
import sys
from time import time_ns

import jax
import numpy as np
import pytest

from feedbax.contracts.canonical_json import canonical_json_v2_bytes
from feedbax.contracts.code_validity import ComputationUnavailable
from feedbax.contracts.computation_validity import ComputationDifferent, ComputationEqual, EvaluatePrefix
from feedbax.contracts.expression_values import CallValue, LiteralValue
from feedbax.contracts.live_evaluate import EVALUATE_DRIVER, ScoredEvaluationPoint, require_evaluation_coverage
from feedbax.contracts.model_input import InitialModelSource, encode_document_input
from feedbax.contracts.receipt_validity import CheckOutcome, CodeReason, ValidityAssessment, ValidityCheckEvidence
from feedbax.contracts.rollout import ResolvedRolloutMember, RolloutParameters
from feedbax.dataset import Dataset, Variable
from feedbax.host.source_roster import SourceRoster
from feedbax.persistence.artifact_custody import ImmutableArtifactBlobProvider
from feedbax.persistence.dataset import publish_dataset
from feedbax.application.composition import load_application
from feedbax.execution.trial_execution import TrialExecution
from feedbax.host.evaluate_computation import EvaluationComputation, RolloutExecutables, drive_replicas
from feedbax.host import evaluate_worker
from feedbax.host.evaluate_worker import _lower_rollouts, _prepare_initial_rollout, _rollouts, _scored_product
from feedbax.validity.admission import decide_validity, observe_differences
from feedbax.validity.computation import compare_computation
from feedbax.validity.recording import WorkerRecorder, numerical_work, record_computation
from feedbax.validity.source import source_catalog
from tests.test_document_train_pair import authored_documents


def test_evaluate_driver_transcript():
    cases = (
        (None, (("0", 4),), 37, 2),
        (6, (("south", 4), ("north/1", 7), ("west", 8)), 83, 4),
        (8, (("west", 8), ("south", 4), ("north/1", 7)), 83, 1),
        (10, (("south", 4),), 83, 1),
        (12, (("south", 4),), 83, 0),
    )
    transcripts = []
    for coordinate, replicas, seed, stop_after in cases:
        calls = []

        def record(slot, operand, key):
            calls.append([slot, operand, key])
            return operand ** 2 + key + len(calls)

        products = drive_replicas(
            coordinate, replicas, seed=seed, call=record, stopped=lambda: len(calls) == stop_after,
        )
        transcripts.append({"calls": calls, "products": list(products) if products is not None else None})
    transcript = canonical_json_v2_bytes(transcripts)
    assert int(sha256(transcript).hexdigest()[:13], 16) == EVALUATE_DRIVER.revision


@pytest.fixture
def initial_rollout(tmp_path):
    _, model, task, objective = authored_documents()
    objective = objective.model_copy(update={
        "terms": tuple(term.model_copy(update={"weight": LiteralValue(value=2)}) for term in objective.terms),
    })
    store = ImmutableArtifactBlobProvider(tmp_path / "store")
    definition = {
        "subject": InitialModelSource(document=model, seed=19),
        "task": task,
        "objective": objective,
    }
    refs = {}
    for role, document in definition.items():
        refs[role], raw = encode_document_input(document)
        store.store_object(raw)
    params = RolloutParameters(
        trials=task.condition_ids[:2], seed=29, trial_policy="paired", variation=(),
        input_bindings={"task-input": "target_pos"},
    )

    def prepare(*, objective_override=None):
        selected = refs
        if objective_override is not None:
            ref, raw = encode_document_input(objective_override)
            store.store_object(raw)
            selected = refs | {"objective": ref}
        return _prepare_initial_rollout(selected, params, registry=load_application().bundle, store=store)

    return store, params, prepare


def test_recorded_trial_matches_preview_and_its_weighted_steps_reduce(initial_rollout, tmp_path):
    """The shelf, reported score and Weight's curve align with the same trial clock."""
    from fastapi.testclient import TestClient

    from feedbax.contracts.trial_record import TrialRecord
    from feedbax.execution.trial_record import aggregate_trials
    from feedbax.persistence.dataset import open_stored_dataset
    from feedbax.persistence.evaluation_trial_records import RecordedTrialLayout, read_trial_record
    from feedbax.web.app import create_app

    store, params, prepare = initial_rollout
    _, model, task, objective = authored_documents()
    ramp = CallValue(function="ramp", args=(LiteralValue(value=1), LiteralValue(value=3)))
    objective = objective.model_copy(update={
        "terms": tuple(term.model_copy(update={"weight": ramp}) for term in objective.terms),
    })
    result = _rollouts(
        prepare(objective_override=objective), params, lambda: False, update_axis=None,
        computation=EvaluationComputation(store), executables=RolloutExecutables(),
    )
    assert result is not None
    dataset, _, *_ = result
    saved = publish_dataset(store, dataset)
    payload = {
        "model": model.model_dump(mode="json"),
        "task": task.model_dump(mode="json"),
        "objective": objective.model_dump(mode="json"),
        "input_bindings": params.input_bindings,
        "seed": params.seed,
    }
    with TestClient(create_app(projects_dir=tmp_path / "projects")) as client:
        def sample(options):
            response = client.post("/api/execution/task-trials/sample", json=payload | options)
            assert response.status_code == 200, response.text
            return [TrialRecord.model_validate(row) for row in response.json()["drawn"]["trials"]]

        previews = sample({"evaluation_trials": list(params.trials)})
        (nominal,) = sample({"nominal": True})

    def check_weight_on_clock(trial):
        (move,) = [cue for cue in trial.timeline if cue.epoch == "move"]
        assert move.start > 0
        weight = trial.terms["effort"].weights
        assert weight is not None and len(weight) == dataset.descriptor.dimensions["transition"]
        assert weight[:move.start] == [None] * move.start
        window = move.end - move.start
        assert weight[move.start:move.end] == pytest.approx([1 + 2 * step / window for step in range(window)])
        assert weight[move.end:] == [None] * (len(weight) - move.end)

    check_weight_on_clock(nominal)
    with open_stored_dataset(store, saved, verify=True) as recorded:
        layout = RecordedTrialLayout.model_validate(recorded.descriptor.attrs["trial_record"])
        values, trials = [], []
        for index, trial in enumerate(previews):
            selection = (0, index)
            restored = read_trial_record(recorded, trial.id, selection, layout)
            assert restored.timeline == trial.timeline
            assert restored.variables == trial.variables
            assert restored.present == trial.present
            check_weight_on_clock(trial)
            check_weight_on_clock(restored)
            terms = restored.terms["effort"]
            assert terms.covered and any(terms.steps) and not all(terms.steps)
            assert terms.values is not None
            (samples,) = restored.observations.values()
            oracle = np.sum(np.asarray(samples) ** 2, axis=-1)
            np.testing.assert_allclose(
                np.asarray(terms.values)[terms.steps], oracle[terms.steps] * np.asarray(terms.weights)[terms.steps].astype(float),
                atol=1e-7,
            )
            measured = np.asarray(terms.values)[terms.steps].mean()
            expected = recorded.data("objective/trial_values/effort", selection).item()
            assert measured == pytest.approx(expected)
            assert terms.value == pytest.approx(expected)
            values.append(measured)
            trials.append(restored)
        expected = recorded.data("objective/terms/effort").item()
        assert np.mean(values) == pytest.approx(expected)
    combined = aggregate_trials(trials, dataset.descriptor.dimensions["transition"]).terms["effort"]
    assert combined.value == pytest.approx(expected)
    assert combined.covered == 1.0 and combined.percentiles is not None
    assert any(band is None for band in combined.percentiles)


def test_measures_share_post_step_velocity_and_record_its_origin_once(tmp_path):
    from feedbax.compiler.model_signals import boundary_signal_port
    from feedbax.contracts.expression_values import LENGTH, TIME, NameValue, ValueBase, ValueType
    from feedbax.contracts.model_address import ModelAddress
    from feedbax.contracts.model_document import ModelDocument
    from feedbax.contracts.objective_document import ObjectiveDocument, ObjectiveTerm
    from feedbax.contracts.semantic_values import semantic_key
    from feedbax.contracts.task_document import TaskDocument
    from feedbax.examples import load_shipped_projects
    from feedbax.objectives.document import ObjectiveUse, compile_objective
    from feedbax.persistence.dataset import open_stored_dataset
    from feedbax.persistence.evaluation_trial_records import RecordedTrialLayout, read_trial_record

    project = load_shipped_projects()["point-mass-reaching"]
    raw = project.documents["point-mass"].payload
    raw["ports"]["outputs"] = [
        {"id": name, "name": name, "type": {}} for name in ("velocity", "effector")
    ]
    graph = raw["interior"]
    graph["components"]["observed-velocity"] = {
        "kind": "leaf", "name": "Observed velocity", "type": "feedbax.SelectState", "fields": {"paths": ["vel"]},
    }
    effector = {"owner": "component", "component": "plant", "port": "effector"}
    graph["wires"] = {
        **{key: wire for key, wire in graph["wires"].items() if wire["target"]["owner"] != "outputs"},
        "observation-select": {
            "source": effector,
            "target": {"owner": "component", "component": "observed-velocity", "port": "state"},
        },
        "observation-velocity": {
            "source": {"owner": "component", "component": "observed-velocity", "port": "output"},
            "target": {"owner": "outputs", "port": "velocity"},
        },
        "observation-effector": {"source": effector, "target": {"owner": "outputs", "port": "effector"}},
    }
    model = ModelDocument.model_validate(raw)
    task = TaskDocument.model_validate(project.documents["simple-reaches-task"].payload)
    root = boundary_signal_port((), "outputs", "velocity")
    structured = ModelAddress.model_validate({
        "composites": [], "element": {
            "kind": "field", "parent": {
                "kind": "port", "end": {"owner": "outputs", "port": "effector"},
            }, "path": ["vel"],
        },
    })
    interior = ModelAddress.model_validate({
        "composites": [], "element": {
            "kind": "field", "parent": {
                "kind": "port", "end": {
                    "owner": "component", "component": "plant", "port": "effector", "direction": "output",
                },
            }, "path": ["vel"],
        },
    })
    velocity2 = ValueType(base=ValueBase.REAL, dimensions=LENGTH.combine(TIME, -1).raised(2))
    objective = ObjectiveDocument(id="velocity-aliases", terms=tuple(
        ObjectiveTerm(
            id=name, measure_type=velocity2,
            measure=CallValue(function="sqnorm", args=(NameValue(reference=address.to_text()),)),
        )
        for name, address in (("root", root), ("structured", structured), ("interior", interior))
    ))
    store = ImmutableArtifactBlobProvider(tmp_path / "store")
    refs = {}
    for role, document in {
        "subject": InitialModelSource(document=model, seed=19), "task": task, "objective": objective,
    }.items():
        refs[role], blob = encode_document_input(document)
        store.store_object(blob)
    params = RolloutParameters(
        trials=task.condition_ids[:2], seed=29, trial_policy="paired", variation=(),
        input_bindings={"target_pos": "target_pos"},
    )
    bundle = load_application().bundle
    prepared = _prepare_initial_rollout(refs, params, registry=bundle, store=store)
    scope = prepared.paired.signal_scope
    assert scope.paths[structured.to_text()] == ("effector", "vel")
    assert scope.paths[interior.to_text()][-2:] == ("effector", "vel")
    (problem,) = compile_objective(
        objective, tuple(name for name in scope.names if name.id != interior.to_text()),
        bundle.expression_functions, task=prepared.paired.execution.task, use=ObjectiveUse.INHERITED,
    )
    assert problem.path == ("terms", "interior", "measure", "args", 0)
    assert problem.model_address == interior
    result = _rollouts(
        prepared, params, lambda: False, update_axis=None,
        computation=EvaluationComputation(store), executables=RolloutExecutables(),
    )
    assert result is not None
    saved = publish_dataset(store, result[0])
    addr = ModelAddress.at_state((), "plant", "effector/velocity").to_text()
    field = semantic_key("model", "state", ("plant",), "effector/velocity")
    with open_stored_dataset(store, saved, verify=True) as recorded:
        layout = RecordedTrialLayout.model_validate(recorded.descriptor.attrs["trial_record"])
        assert layout.observations == {addr: field}
        assert not any(name.startswith("/v1/model/output/") for name in recorded.descriptor.variables)
        for index, trial in enumerate(params.trials):
            selection = (0, index)
            restored = read_trial_record(recorded, trial, selection, layout)
            raw_velocity = recorded.data(field, selection)
            assert np.any(raw_velocity[1:] != raw_velocity[:-1])
            np.testing.assert_array_equal(restored.observations[addr], raw_velocity[1:])
            oracle = np.sum(raw_velocity[1:] ** 2, axis=-1)
            for term in restored.terms.values():
                np.testing.assert_allclose(term.values, oracle, atol=1e-7)


def test_managed_rollout_relowers_without_compiling_and_detects_changed_scoring(initial_rollout, monkeypatch):
    store, params, prepare = initial_rollout
    execution = EvaluationComputation(store)
    result = _rollouts(
        prepare(), params, lambda: False, update_axis=None, computation=execution, executables=RolloutExecutables(),
    )
    assert result is not None
    dataset, total, *_ = result
    assert int(dataset.data("objective/coverage/effort").item()) == 2
    assert total == dataset.data("objective/total").item()
    np.testing.assert_allclose(dataset.data("objective/trial_loss").mean(axis=-1), dataset.data("objective/total"))
    previous = execution.snapshot()
    assert previous is not None and isinstance(previous.prefix, EvaluatePrefix)
    assert tuple(output.slot for output in previous.prefix.outputs) == ("/initial/0",)

    def recheck():
        prepared = prepare()
        present = EvaluationComputation(store)
        with monkeypatch.context() as probe:
            def refuse_compile(*args, **kwargs):
                pytest.fail("Validity check compiled the rollout")

            probe.setattr(jax.stages.Lowered, "compile", refuse_compile)
            _lower_rollouts(prepared, lambda: False, consume=lambda _, call: present.observe(call))
        assert (current := present.snapshot()) is not None
        return compare_computation(previous, current, read_blob=store.get_object)

    assert isinstance(recheck(), ComputationEqual)
    evaluate = TrialExecution.evaluate

    def changed_score(self, *args, **kwargs):
        result = evaluate(self, *args, **kwargs)
        assert result.objective is not None
        return result._replace(objective=result.objective._replace(loss=result.objective.loss + 1))

    with monkeypatch.context() as probe:
        probe.setattr(TrialExecution, "evaluate", changed_score)
        difference = recheck()
    assert isinstance(difference, ComputationDifferent)
    assert [(item.kind, item.slots) for item in difference.changes] == [("program", ("/initial/0",))]

    class OpaqueStateView:
        def __call__(self, state):
            return state

    prepared = prepare()
    graph = replace(prepared.graph, state_view_fn=OpaqueStateView())
    opaque = EvaluationComputation(store)
    outcome = _rollouts(
        replace(prepared, graph=graph), params, lambda: False, update_axis=None,
        computation=opaque, executables=RolloutExecutables(),
    )
    assert outcome is not None and outcome[1] == total
    unavailable = opaque.snapshot()
    assert isinstance(unavailable, ComputationUnavailable)
    assert unavailable.reason == "arguments"
    require_evaluation_coverage(unavailable, ("/initial/0",))


def test_equal_rollout_check_preserves_protected_wrapper_staleness(tmp_path, initial_rollout, monkeypatch):
    store, params, prepare = initial_rollout
    project = tmp_path / "project"
    project.mkdir()
    score = project / "scope_score.py"
    wrapper = project / "scope_product.py"
    score.write_text("def run(*args, **kwargs):\n    return delegate(*args, **kwargs)\n")
    wrapper.write_text("def run(*args):\n    return delegate(*args)\n")

    def load(path, delegate):
        spec = spec_from_file_location(path.stem, path)
        module = module_from_spec(spec)
        module.delegate = delegate
        monkeypatch.setitem(sys.modules, spec.name, module)
        spec.loader.exec_module(module)
        return module.run

    original_score = TrialExecution.evaluate
    original_product = evaluate_worker._rollout_product
    execution = EvaluationComputation(store)
    with SourceRoster(3, "test Evaluate source") as roster:
        roster.enable()
        recorder = WorkerRecorder(roster, boot_ns=time_ns())
        with recorder.monitor():
            recorder.bind_project(project)
            monkeypatch.setattr(TrialExecution, "evaluate", load(score, original_score))
            monkeypatch.setattr(evaluate_worker, "_rollout_product", load(wrapper, original_product))
            with recorder.delivery("evaluate", store.root, tmp_path / "scratch"):
                with numerical_work():
                    prepared = prepare()
                result = _rollouts(
                    prepared, params, lambda: False, update_axis=None,
                    computation=execution, executables=RolloutExecutables(),
                )
                assert result is not None
                record_computation(execution.snapshot)
                recorded = recorder.snapshot(store)

    receipt = store.store_object(b"Evaluate receipt identity for the validity decision")
    trace = recorded.trace
    registry = {(binding.family, binding.requested_key): binding for binding in trace.registry}
    score.write_text("def run(*args, **kwargs):\n    scored = delegate(*args, **kwargs)\n    return scored\n")
    monkeypatch.setattr(TrialExecution, "evaluate", load(score, original_score))

    def compare():
        conditions, files = observe_differences(
            (trace,), catalog=source_catalog(project), registry=registry, environment=trace.environment,
            read_object=store.get_object,
        )
        assert not files
        replay = EvaluationComputation(store)
        _lower_rollouts(prepare(), lambda: False, consume=lambda _, call: replay.observe(call))
        former, current = trace.computation, replay.snapshot()
        assert isinstance(compare_computation(former, current), ComputationEqual)
        assessment = ValidityAssessment(
            receipt=receipt, conditions=conditions,
            evidence=ValidityCheckEvidence(code=CheckOutcome(outcome="equal")), assessed_at=datetime.now(UTC),
        )
        return decide_validity(receipt, conditions=conditions, records=(assessment,), code_checkable=True)

    assert compare().state == "current"
    wrapper.write_text(
        "def run(*args):\n    dataset, total, member, objective = delegate(*args)\n"
        "    return dataset, total + 1, member, objective\n"
    )
    checked = compare()
    assert checked.state == "stale"
    (reason,) = checked.reasons
    assert isinstance(reason, CodeReason)
    assert [change.path.path for change in reason.differences] == ["scope_product.py"]


def test_live_checkpoint_preserves_the_replica_coordinate_and_mean(tmp_path):
    store = ImmutableArtifactBlobProvider(tmp_path)
    params = RolloutParameters(trials=("0",), seed=17, trial_policy="paired", variation=())
    dataset = Dataset.from_arrays(
        {"objective/total": Variable(np.asarray([[[1.0], [3.0]]]), ("update", "replicate", "condition"))},
        coords={
            "update": Variable(np.array([4]), ("update",)),
            "replicate": Variable(np.array(["A", "B"]), ("replicate",)),
            "trial": Variable(np.array(["0"]), ("trial",)),
        }, schema_id=params.states_schema,
    )
    states = publish_dataset(store, dataset)
    member = ResolvedRolloutMember(
        coordinate=4, model_tree="1" * 64, model="2" * 64, task="3" * 64, trials=params.trials,
        replica_ids=("A", "B"),
    )
    point = ScoredEvaluationPoint(
        states=states, member=member, objective="4" * 64, total=2.0, evidence=states.object_ref,
    )
    with ExitStack() as opened:
        retained, mean = _scored_product(opened, store, point, 4, params)
        assert retained.data("replicate").tolist() == ["A", "B"]
        assert mean == 2.0
        with pytest.raises(ValueError, match="summary differs"):
            _scored_product(opened, store, point.model_copy(update={"total": 3.0}), 4, params)
