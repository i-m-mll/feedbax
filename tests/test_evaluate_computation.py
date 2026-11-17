"""Observable computation and driver evidence for managed rollout scoring."""

from contextlib import ExitStack
from hashlib import sha256

import jax
import numpy as np
import pytest

from feedbax.contracts.canonical_json import canonical_json_v2_bytes
from feedbax.contracts.computation_validity import ComputationDifferent, ComputationEqual, EvaluatePrefix
from feedbax.contracts.live_evaluate import EVALUATE_DRIVER, ScoredEvaluationPoint
from feedbax.contracts.model_input import InitialModelSource, encode_document_input
from feedbax.contracts.rollout import ResolvedRolloutMember, RolloutParameters
from feedbax.dataset import Dataset, Variable
from feedbax.persistence.artifact_custody import ImmutableArtifactBlobProvider
from feedbax.persistence.dataset import publish_dataset
from feedbax.plugins.composition import load_application
from feedbax.runtime.trial_execution import TrialExecution
from feedbax.training.evaluate_computation import EvaluationComputation, RolloutExecutables, drive_replicas
from feedbax.training.evaluate_worker import _lower_rollouts, _prepare_initial_rollout, _rollouts, _scored_product
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
