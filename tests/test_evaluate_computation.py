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

    def prepare():
        return _prepare_initial_rollout(refs, params, registry=load_application().bundle, store=store)

    return store, params, prepare


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
