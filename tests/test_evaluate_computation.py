"""Observable computation and driver evidence for managed rollout scoring."""

from hashlib import sha256

import jax
import pytest

from feedbax.contracts.canonical_json import canonical_json_v2_bytes
from feedbax.contracts.computation_validity import ComputationDifferent, ComputationEqual, EvaluatePrefix
from feedbax.contracts.live_evaluate import EVALUATE_DRIVER
from feedbax.contracts.model_input import InitialModelSource, encode_document_input
from feedbax.contracts.rollout import RolloutParameters
from feedbax.persistence.artifact_custody import ImmutableArtifactBlobProvider
from feedbax.plugins.composition import load_application
from feedbax.runtime.trial_execution import TrialExecution
from feedbax.training.evaluate_computation import EvaluationComputation, RolloutExecutables, drive_replicas
from feedbax.training.evaluate_worker import _lower_rollouts, _prepare_initial_rollout, _rollouts
from feedbax.validity.computation import compare_computation
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


def test_managed_rollout_relowers_without_compiling_and_detects_changed_scoring(tmp_path, monkeypatch):
    _, model, task, objective = authored_documents()
    store = ImmutableArtifactBlobProvider(tmp_path)
    definition = {
        "subject": InitialModelSource(model_name="reach-model", document=model, seed=19),
        "task": task,
        "objective": objective,
    }
    refs = {}
    for role, document in definition.items():
        refs[role], raw = encode_document_input(document)
        store.store_object(raw)
    params = RolloutParameters(
        trials=task.condition_ids[:2], seed=29, trial_policy="paired", variation=(),
        input_overrides={"reach_target": "target_pos"},
    )

    def prepare():
        return _prepare_initial_rollout(refs, params, registry=load_application().bundle, store=store)

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

    monkeypatch.setattr(TrialExecution, "evaluate", changed_score)
    difference = recheck()
    assert isinstance(difference, ComputationDifferent)
    assert [(item.kind, item.slots) for item in difference.changes] == [("program", ("/initial/0",))]
