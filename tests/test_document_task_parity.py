"""Step 1 exercises the document runtime beside the existing class entry points."""

import time

from dataclasses import replace

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
import pytest
from scipy.stats import ks_2samp

from feedbax.compiler import GraphDocument, compile_graph
from feedbax.compiler.model_document import ResolvedModelInterface, lower_model_document
from feedbax.component_registry import ComponentRegistry
from feedbax.contracts.model_address import ModelAddress
from feedbax.contracts.model_document import ModelDocument
from feedbax.contracts.expression_values import (
    LENGTH,
    TIME,
    Level,
    ScopedValue,
    Site,
    ValueBase,
    ValueType,
)
from feedbax.expression_builtins import register_builtins
from feedbax.expression_registry import FunctionRegistry
from feedbax.expression_text import ParsedField, parse
from feedbax.objectives.document import (
    ObjectiveDocument,
    ObjectiveExecutor,
    ObjectiveInputs,
    ObjectiveTerm,
    Reduction,
    compile_objective,
)
from feedbax.tasks.document import (
    Cell,
    DocumentTask,
    Epoch,
    EpochEnd,
    Link,
    Signal,
    TaskDocument,
    TaskParameters,
    ValidationTrial,
    Variable,
    compile_task,
)
from feedbax.runtime.graph import GraphTraceRequest
from feedbax.runtime.model_reset import ModelResetError
from feedbax.tasks.task import (
    DelayedReaches,
    SimpleReaches,
    Stabilization,
    gen_epoch_lengths,
)
from tests.test_model_document_plan import _model_project
from feedbax.xabdeef.losses import delayed_reach_loss, simple_reach_loss
from feedbax.xabdeef.models import point_mass_nn

PERIOD = 0.01
STEPS = 80
WORKSPACE = jnp.array([[-0.4, -0.3], [0.5, 0.45]])
POINT = ValueType(base=ValueBase.REAL, shape=(2,), dimensions=LENGTH)
REAL = ValueType(base=ValueBase.REAL)
TIME_TYPE = ValueType(base=ValueBase.REAL, dimensions=TIME)
BOOLEAN = ValueType(base=ValueBase.BOOLEAN)


PORTLESS = ResolvedModelInterface(PERIOD, (), (), (), ())


def reset_position(interface):
    return next(
        field.address for field in interface.reset.fields
        if field.address.element.path == ("reset", "effector", "position")
    )


def _model_payload(*, curl=False, writable_gru=False, rename=False):
    payload = _model_project()["documents"]["reach-model"]["payload"]
    payload["step_seconds"] = PERIOD
    inputs = payload["ports"]["inputs"]
    inputs[0]["type"]["unit"] = "m"
    payload["ports"]["outputs"] = [
        {"id": "command", "name": "command", "type": {"shape": [2]}},
    ]
    payload["interior"]["wires"]["effector-wire"] = {
        "source": {"owner": "component", "component": "wrapper", "port": "force"},
        "target": {"owner": "outputs", "port": "command"},
    }
    if curl:
        payload["interior"]["components"]["curl"] = {
            "kind": "leaf", "name": "Curl", "type": "feedbax.CurlField",
        }
        payload["interior"]["wires"]["plant-wire"]["target"] = {"owner": "component", "component": "curl", "port": "force"}
        payload["interior"]["wires"]["curl-drive"] = {
            "source": {"owner": "component", "component": "curl", "port": "force"},
            "target": {"owner": "component", "component": "plant", "port": "force"},
        }
        payload["interior"]["wires"]["curl-feedback"] = {
            "source": {"owner": "component", "component": "plant", "port": "effector"},
            "target": {"owner": "component", "component": "curl", "port": "effector"},
        }
    if writable_gru:
        wrapper = payload["interior"]["components"]["wrapper"]
        cell_owner = wrapper["interior"]["components"]["controller"]
        controller = cell_owner["interior"]
        del controller["wires"]["carry"]
        controller["components"]["cell"]["fields"]["stateful"] = True
        hidden_port = {"id": "hidden", "name": "hidden", "type": {}}
        payload["ports"]["outputs"].append(hidden_port)
        for owner, name, output in (
            (payload, "wrapper", "hidden"),
            (wrapper, "controller", "hidden"),
            (cell_owner, "cell", "output"),
        ):
            if owner is not payload:
                owner["ports"]["outputs"].append(hidden_port)
            owner["interior"]["wires"]["hidden-output"] = {
                "source": {"owner": "component", "component": name, "port": output},
                "target": {"owner": "outputs", "port": "hidden"},
            }
        controller["wires"]["decode"]["attachments"] = [{
            "kind": "delay", "id": "read-delay", "delay": {"value": 2, "unit": "st"},
            "initial": {"kind": "name", "reference": ModelAddress.model_validate({
                "document": "reach-model", "composites": ["wrapper", "controller"],
                "element": {"kind": "port", "end": {
                    "owner": "component", "component": "cell", "port": "output", "direction": "output",
                }},
            }).to_text()},
        }]
    if rename:
        payload["interior"]["components"]["plant"]["name"] = "Renamed plant"
        payload["interior"]["components"]["wrapper"]["name"] = "Renamed wrapper"
    return payload


def _pair_model(**options):
    model = ModelDocument.model_validate(_model_payload(**options))
    registry = ComponentRegistry()
    lowered = lower_model_document("reach-model", model, registry)
    source = lowered.graph.input_bindings["reach_target"]
    prototypes = {source: jax.ShapeDtypeStruct((2,), jnp.float32)}
    interface = lowered.interface("reach-model", model, registry, input_prototypes=prototypes)
    graph = compile_graph(
        GraphDocument(graph=lowered.graph),
        registry,
        input_prototypes=prototypes,
        initialization_key=jr.key(4),
    ).graph
    return interface, graph


@pytest.fixture(scope="module")
def paired_model():
    return _pair_model()


@pytest.fixture
def functions():
    registry = FunctionRegistry()
    register_builtins(registry)
    registry.seal()
    return registry


def name(identifier, value_type, level=Level.STEP):
    return ScopedValue(id=identifier, name=identifier, type=value_type, level=level)


def expression(text, scope=()):
    result = parse(text, Site(expected=REAL, finest=Level.STEP, names=tuple(scope)))
    assert isinstance(result, ParsedField), result
    return result.value


def example(kind, functions, *, zero=False, interface=PORTLESS, old=None):
    delayed = kind == "delayed"
    initial = kind == "stabilization"
    names = (
        name("start", POINT, Level.TRIAL),
        name("target", POINT, Level.TRIAL),
        name("catch", BOOLEAN, Level.TRIAL),
    )

    def variable(index, text):
        item = names[index]
        return Variable(id=item.id, name=item.name, type=item.type, value=expression(text, names))

    sampled = "[U(-0.4, 0.5), U(-0.3, 0.45)]"
    variables = [variable(0, sampled), variable(1, "start" if initial else sampled)]
    if delayed:
        variables.append(variable(2, "B(0.5)"))
        first = expression("0 st" if zero else "U(5 st, 14 st)")
        epochs = (
            Epoch(id="hold", name="hold", length=first, max_steps=14),
            Epoch(id="cue", name="cue", length=expression("U(10 st, 19 st)"), max_steps=19),
            Epoch(id="move", name="move", length=None),
        )
    else:
        epochs = (Epoch(id="move", name="move", length=expression(f"{STEPS} st")),)

    def cell(start, end, text):
        return Cell(first_epoch=start, last_epoch=end, value=expression(text, names))

    def signal(identifier, value_type, *cells):
        return Signal(id=identifier, name=identifier, type=value_type, cells=cells)

    first = epochs[0].id
    target_cells = (
        (cell("hold", "cue", "start"), cell("move", "move", "where(catch, start, target)"))
        if delayed
        else (cell(first, "move", "target"),)
    )
    signals = [
        signal("target_pos", POINT, cell("cue" if delayed else first, "move", "target")),
        signal(
            "target_vel",
            ValueType(base=ValueBase.REAL, shape=(2,), dimensions=LENGTH.combine(TIME, -1)),
            cell("cue" if delayed else first, "move", "[0, 0]"),
        ),
        signal("goal", POINT, *target_cells),
    ]
    if delayed:
        signals.extend(
            (
                signal(
                    "fixation",
                    REAL,
                    cell("hold", "cue", "1"),
                    cell("move", "move", "where(catch, 1, 0)"),
                ),
                signal("target_on", REAL, cell("cue", "move", "1")),
            )
        )
    doc = TaskDocument(
        id=kind,
        variables=tuple(variables),
        epochs=epochs,
        signals=tuple(signals),
        links=(Link(model_field=reset_position(interface).to_text(), variable="start"),) if interface.reset.fields else (),
        fixed_steps=STEPS,
        validation_trials=validation_rows(kind, old) if old else (ValidationTrial(),),
    )
    task = compile_task(
        doc,
        functions,
        model=interface,
        input_overrides={"reach_target": "target_pos"} if interface.inputs else None,
    )
    assert isinstance(task, DocumentTask), task
    return doc, task


def legacy(kind):
    common = dict(n_steps=STEPS + 1, workspace=WORKSPACE, eval_grid_n=2)
    if kind == "delayed":
        return DelayedReaches(
            **common, loss_func=delayed_reach_loss(), catch_metadata_policy="flag"
        )
    cls = Stabilization if kind == "stabilization" else SimpleReaches
    return cls(**common, loss_func=simple_reach_loss(), epoch_name="move")


def validation_rows(kind, old):
    size = old.eval_grid_n
    bounds = np.asarray(old.workspace)
    stabilization = kind == "stabilization"
    ticks = [
        np.linspace(lo, hi, size if stabilization else size + 2, dtype=np.float32)
        for lo, hi in bounds.T
    ]
    if not stabilization:
        ticks = [axis[1:-1] for axis in ticks]
    start = np.stack(np.meshgrid(*ticks), axis=-1).reshape(-1, 2)
    if stabilization:
        goal = start
    else:
        angles = np.arange(old.eval_n_directions) * (2 * np.pi / old.eval_n_directions)
        offsets = old.eval_reach_length * np.stack((np.cos(angles), np.sin(angles)), axis=-1)
        goal = (start[:, None] + offsets).reshape(-1, 2)
        start = np.repeat(start, old.eval_n_directions, axis=0)
    if kind == "delayed":
        trials_key = jr.split(jr.PRNGKey(old.seed_validation), 3)[0]
        epoch_keys = jax.vmap(lambda key: jr.split(key)[0])(jr.split(trials_key, len(start)))
        lengths = jax.vmap(lambda key: gen_epoch_lengths(key, old.epoch_len_ranges))(epoch_keys)
    rows = []
    for i, (a, b) in enumerate(zip(start, goal, strict=True)):
        variables = {"start": tuple(map(float, a)), "target": tuple(map(float, b))}
        epoch_steps = {}
        if kind == "delayed":
            variables["catch"] = False
            epoch_steps = dict(zip(("hold", "cue"), map(int, lengths[i]), strict=True))
        rows.append(ValidationTrial(variables=variables, epoch_steps=epoch_steps))
    return tuple(rows)


def overrides(kind, old_trials):
    start = old_trials.inits["mechanics.effector"].pos
    target = (
        old_trials.inputs.effector_target
        if kind != "simple"
        else old_trials.inputs["effector_target"]
    )
    given = {"start": start, "target": target.pos[:, -1]}
    epoch_steps = {}
    if kind == "delayed":
        lengths = jnp.diff(old_trials.timeline.epoch_bounds, axis=-1)
        given["catch"] = old_trials.extra["is_catch_trial"]
        epoch_steps = {"hold": lengths[:, 0], "cue": lengths[:, 1]}
    return TaskParameters(variables=given, epoch_steps=epoch_steps)


def objective(kind, task, functions):
    velocity = ValueType(base=ValueBase.REAL, shape=(2,), dimensions=LENGTH.combine(TIME, -1))
    retained = (
        name("hand", POINT),
        name("velocity", velocity),
        name("command", ValueType(base=ValueBase.REAL, shape=(2,))),
        name("activity", ValueType(base=ValueBase.REAL, shape=(4,))),
    )
    scope = (*retained, *task.interface.variables, *task.interface.signals, name("u", REAL))
    terms = []

    def add(identifier, measure, dimensions=REAL.dimensions, weight="1", reduction=Reduction.SUM):
        terms.append(
            ObjectiveTerm(
                id=identifier,
                measure=expression(measure, scope),
                measure_type=ValueType(base=ValueBase.REAL, dimensions=dimensions),
                weight=expression(weight, scope),
                reduction=reduction,
            )
        )

    if kind == "delayed":
        add("effector_fixation", "sqnorm(hand - goal) * fixation", LENGTH.raised(2))
    discount = "1"
    if kind == "simple":
        x = f"(u + {1 / STEPS})"
        discount = " * ".join([x] * 6)
    add("effector_position", "sqnorm(hand - goal)", LENGTH.raised(2), discount)
    add(
        "effector_final_velocity",
        "sqnorm(velocity)",
        velocity.dimensions.raised(2),
        reduction=Reduction.LAST,
    )
    add("nn_output", "sqnorm(command)", weight="0.0001" if kind == "delayed" else "0.00001")
    add("nn_hidden", "sqnorm(activity)", weight="0.00001")
    executor = compile_objective(
        ObjectiveDocument(id=kind, terms=tuple(terms)), scope[:-1], functions, task=task
    )
    assert isinstance(executor, ObjectiveExecutor), executor
    return executor


def vectorized_signals(kind, trial):
    steps = jnp.arange(STEPS)
    goal = trial.variables["target"][:, None, :]
    target = jnp.broadcast_to(goal, (goal.shape[0], STEPS, 2))
    if kind != "delayed":
        return {"target_pos": target, "target_vel": jnp.zeros_like(target), "goal": target}
    cue = trial.epochs.starts[:, 1, None]
    move = trial.epochs.starts[:, 2, None]
    catch = trial.variables["catch"][:, None]
    fixation = (steps < move) | catch
    visible = steps >= cue
    return {
        "target_pos": jnp.where(visible[..., None], target, 0),
        "target_vel": jnp.zeros_like(target),
        "goal": jnp.where(fixation[..., None], trial.variables["start"][:, None, :], target),
        "fixation": fixation,
        "target_on": visible,
    }


@pytest.mark.parametrize("kind", ["simple", "delayed", "stabilization"])
def test_document_trial_and_objective_match_legacy(kind, functions):
    old = legacy(kind)
    doc, task = example(kind, functions, old=old)
    assert TaskDocument.model_validate_json(doc.model_dump_json()) == doc
    run_key = jr.key(1)
    keys = jr.split(run_key, 250)
    old_trials = jax.jit(jax.vmap(old.get_train_trial))(keys)
    given = overrides(kind, old_trials)
    batch = jax.jit(jax.vmap(lambda key, parameters: task.ahead(parameters, run_key, key)))
    trial = batch(keys, given)
    expected_inputs = old_trials.inputs
    effector = (
        expected_inputs.effector_target if kind != "simple" else expected_inputs["effector_target"]
    )
    np.testing.assert_array_equal(trial.inputs["target_pos"], effector.pos)
    np.testing.assert_array_equal(trial.inputs["target_vel"], effector.vel)
    np.testing.assert_array_equal(
        trial.inputs["goal"], old_trials.targets["mechanics.effector.pos"].value
    )
    np.testing.assert_array_equal(
        trial.variables["start"], old_trials.inits["mechanics.effector"].pos
    )
    np.testing.assert_array_equal(trial.epochs.starts, old_trials.timeline.epoch_bounds[:, :-1])
    np.testing.assert_array_equal(trial.epochs.ends, old_trials.timeline.epoch_bounds[:, 1:])
    assert (trial.epochs.outcomes == EpochEnd.LENGTH).all()
    if kind == "delayed":
        np.testing.assert_array_equal(trial.inputs["fixation"], expected_inputs.hold[..., 0])
        np.testing.assert_array_equal(trial.inputs["target_on"], expected_inputs.target_on[..., 0])
        np.testing.assert_array_equal(trial.variables["catch"], old_trials.extra["is_catch_trial"])
    for signal, expected in vectorized_signals(kind, trial).items():
        np.testing.assert_array_equal(trial.inputs[signal], expected)
    validation = jax.jit(task.validation)(TaskParameters(), run_key)
    legacy_validation = old.validation_trials
    target = legacy_validation.targets["mechanics.effector.pos"].value
    assert validation.inputs["goal"].shape == (old.n_validation_trials, STEPS, 2)
    np.testing.assert_allclose(validation.inputs["goal"], target, rtol=0, atol=3e-7)
    np.testing.assert_allclose(
        validation.variables["start"],
        legacy_validation.inits["mechanics.effector"].pos,
        rtol=0,
        atol=3e-7,
    )
    np.testing.assert_array_equal(
        validation.epochs.starts, legacy_validation.timeline.epoch_bounds[:, :-1]
    )
    np.testing.assert_array_equal(
        validation.epochs.ends, legacy_validation.timeline.epoch_bounds[:, 1:]
    )
    np.testing.assert_array_equal(
        validation.inputs["goal"], task.validation(TaskParameters(), jr.key(99)).inputs["goal"]
    )

    old_inputs = legacy_validation.inputs
    old_effector = old_inputs["effector_target"] if kind == "simple" else old_inputs.effector_target
    np.testing.assert_allclose(validation.inputs["target_pos"], old_effector.pos, rtol=0, atol=3e-7)
    np.testing.assert_array_equal(validation.inputs["target_vel"], old_effector.vel)
    if kind == "delayed":
        np.testing.assert_array_equal(validation.inputs["fixation"], old_inputs.hold[..., 0])
        np.testing.assert_array_equal(validation.inputs["target_on"], old_inputs.target_on[..., 0])
        assert not validation.variables["catch"].any()

    # Compare objective values on an actual PointMass controller rollout.
    model = point_mass_nn(old, n_steps=STEPS + 1, dt=PERIOD, hidden_size=4, key=jr.key(7))
    states = old.eval_trials(model, old_trials, keys)
    retained = {
        "hand": states.mechanics.effector.pos,
        "velocity": states.mechanics.effector.vel,
        "command": states.efferent.output,
        "activity": states.net.hidden,
    }
    executor = objective(kind, task, functions)
    # The fixation signal has an explicit zero outside the scored epochs.
    results = jax.jit(
        lambda kept: executor.evaluate(
            kept,
            trial,
            run_key=run_key,
            trial_keys=keys,
        )
    )(retained)
    if kind == "simple":
        # The legacy trainer stores this discount once for the batch. Raw vmap
        # broadcasts the shared field, so restore its declared batch axes.
        old_trials = eqx.tree_at(
            lambda trial: trial.targets["mechanics.effector.pos"].discount,
            old_trials,
            old_trials.targets["mechanics.effector.pos"].discount[0],
        )
    expected_loss = old.loss_func(states, old_trials, model)
    tree = executor(retained, trial, ObjectiveInputs(run_key, keys))
    assert tree.total == pytest.approx(float(results.loss), rel=2e-5, abs=1e-6)
    assert results.loss == pytest.approx(float(expected_loss.total), rel=2e-5, abs=1e-6)
    for term, observed in results.terms.items():
        assert int(observed.covered) == keys.shape[0]
        assert float(observed.value) == pytest.approx(
            float(expected_loss[term].total), rel=2e-5, abs=1e-6
        )

    samples = []
    for _ in range(7):
        start = time.perf_counter()
        batch(keys, given).inputs["goal"].block_until_ready()
        samples.append((time.perf_counter() - start) * 1000)
    median = np.median(samples)
    print(f"{kind} batch250 median {median:.3f} ms")


def test_zero_length_and_delayed_distribution(functions):
    _, zero = example("delayed", functions, zero=True)
    key = jr.key(5)
    trial = jax.jit(zero.ahead)(TaskParameters(), key, key)
    assert trial.epochs.starts[:2].tolist() == [0, 0]
    assert trial.epochs.ends[0] == 0
    assert trial.present["target_pos"][-1]

    _, task = example("delayed", functions)
    old = legacy("delayed")
    old_keys = jr.split(jr.key(16), 20_000)
    new_keys = jr.split(jr.key(17), 20_000)

    def old_draw(key):
        trial = old.get_train_trial(key)
        return (
            jnp.diff(trial.timeline.epoch_bounds)[:2],
            trial.extra["is_catch_trial"],
            trial.inits["mechanics.effector"].pos,
            trial.inputs.effector_target.pos[-1],
        )

    def new_draw(key):
        prepared, _ = task.init(TaskParameters(), key, key)
        epochs = prepared.epochs
        return (
            (epochs.ends - epochs.starts)[:2],
            prepared.values["catch"],
            prepared.values["start"],
            prepared.values["target"],
        )

    before = jax.jit(jax.vmap(old_draw))(old_keys)
    after = jax.jit(jax.vmap(new_draw))(new_keys)
    for index, (lo, hi) in enumerate(old.epoch_len_ranges):
        for values in (before[0][:, index], after[0][:, index]):
            np.testing.assert_array_equal(np.unique(values), np.arange(lo, hi))
            proportions = np.bincount(np.asarray(values) - lo) / len(values)
            assert np.max(np.abs(proportions - 1 / (hi - lo))) < 0.01
    assert abs(float(jnp.mean(before[1])) - float(jnp.mean(after[1]))) < 0.02
    for b, a in zip(before[2:], after[2:], strict=True):
        for dimension in range(2):
            assert (
                ks_2samp(np.asarray(b[:, dimension]), np.asarray(a[:, dimension])).statistic < 0.025
            )


def test_objective_windows_reduce_defined_samples_with_history(functions):
    from feedbax.runtime.epoch_windows import EpochRecord

    keys = jr.split(jr.key(3), 3)
    key = keys[0]
    x = name("x", REAL)
    steps = 5
    task = compile_task(
        TaskDocument(
            id="windows",
            variables=(),
            signals=(),
            epochs=(
                Epoch(id="first", name="first", length=expression("1 st")),
                Epoch(id="left", name="left", length=expression("2 st")),
                Epoch(id="right", name="right", length=expression("2 st")),
            ),
        ),
        functions,
        model=PORTLESS,
    )
    assert isinstance(task, DocumentTask), task
    batch = jax.vmap(lambda trial_key: task.ahead(TaskParameters(), key, trial_key))(keys)
    batch = batch._replace(
        epochs=EpochRecord(
            starts=jnp.asarray([[0, 1, 3], [0, -1, -1], [-1, -1, -1]]),
            ends=jnp.asarray([[1, 3, 5], [1, -1, -1], [-1, -1, -1]]),
            outcomes=jnp.asarray([[1, 1, 1], [1, 0, 0], [0, 0, 0]]),
        )
    )
    formula = expression("diff(x)", (x,))
    terms = tuple(
        ObjectiveTerm(
            id=reduction.value,
            measure=formula,
            reduction=reduction,
            epochs=("first", "left", "right"),
        )
        for reduction in Reduction
    ) + (
        ObjectiveTerm(id="left", measure=formula, reduction=Reduction.SUM, epochs=("left",)),
        ObjectiveTerm(id="right", measure=formula, reduction=Reduction.SUM, epochs=("right",)),
        ObjectiveTerm(id="disabled", measure=expression("42"), enabled=False),
    )
    executor = compile_objective(
        ObjectiveDocument(id="difference", terms=terms), (x,), functions, task=task
    )
    assert isinstance(executor, ObjectiveExecutor), executor
    retained = {"x": jnp.broadcast_to(jnp.arange(steps, dtype=float) ** 2, (3, steps))}
    score = jax.jit(
        lambda trace: executor.evaluate(
            trace,
            batch,
            run_key=key,
            trial_keys=keys,
        )
    )
    actual = score(retained)
    assert set(actual.terms) == {"sum", "mean", "last", "max", "left", "right"}
    expected = {"sum": 16.0, "mean": 4.0, "last": 7.0, "max": 7.0, "left": 4.0, "right": 12.0}
    for term, value in actual.terms.items():
        assert int(value.covered) == 1
        assert float(value.value) == expected[term]
    assert actual.terms["left"].value + actual.terms["right"].value == actual.terms["sum"].value
    assert float(actual.loss) == sum(expected.values())
    assert int(score({}).terms["sum"].covered) == 0
    derivative = jax.grad(lambda trace: score({"x": trace}).terms["right"].value)(retained["x"])
    np.testing.assert_array_equal(derivative[0], [0, 0, -1, 0, 1])
    np.testing.assert_array_equal(derivative[1:], 0)


def test_inherited_objective_preserves_term_ids_without_observing_missing_values(functions):
    from feedbax.objectives.document import ObjectiveUse

    x, y = name("x", REAL), name("y", REAL)
    task = compile_task(
        TaskDocument(id="replacement", variables=(), signals=(), epochs=(Epoch(id="probe", name="Probe"),),
                     fixed_steps=2),
        functions, model=PORTLESS,
    )
    assert isinstance(task, DocumentTask), task
    document = ObjectiveDocument(id="inherited", terms=(
        ObjectiveTerm(id="covered", measure=expression("x", (x,))),
        ObjectiveTerm(id="missing-measure", measure=expression("y", (y,))),
        ObjectiveTerm(id="missing-weight", measure=expression("x", (x,)), weight=expression("y", (y,))),
        ObjectiveTerm(id="missing-epoch", measure=expression("x", (x,)), epochs=("probe", "move")),
    ))
    strict = compile_objective(document, (x,), functions, task=task)
    assert not isinstance(strict, ObjectiveExecutor)
    assert {problem.path[:2] for problem in strict} == {
        ("terms", "missing-measure"), ("terms", "missing-weight"), ("terms", "missing-epoch"),
    }
    scorer = compile_objective(document, (x,), functions, task=task, use=ObjectiveUse.INHERITED)
    assert isinstance(scorer, ObjectiveExecutor), scorer
    assert [item.id for item in scorer.names] == ["x"]
    key = jr.key(0)
    batch = task.validation(TaskParameters(), key)
    result = scorer.evaluate({"x": jnp.asarray([[1.0, 3.0]])}, batch, run_key=key, trial_keys=key[None])
    assert float(result.loss) == 2
    assert {name: int(value.covered) for name, value in result.terms.items()} == {
        "covered": 1, "missing-measure": 0, "missing-weight": 0, "missing-epoch": 0,
    }


def test_temporal_variable_can_leave_a_cell_absent(functions):
    time_name = name("time", TIME_TYPE)
    velocity = name("velocity", REAL)
    document = TaskDocument(
        id="coverage",
        variables=(
            Variable(
                id="velocity",
                name="velocity",
                type=REAL,
                value=expression("ddt(time)", (time_name,)),
            ),
        ),
        epochs=(
            Epoch(id="before", name="before", length=expression("2 st")),
            Epoch(id="after", name="after", length=expression("2 st")),
        ),
        signals=(
            Signal(
                id="seen",
                name="seen",
                type=REAL,
                cells=(
                    Cell(
                        first_epoch="before",
                        last_epoch="after",
                        value=expression("velocity", (velocity,)),
                    ),
                ),
            ),
        ),
    )
    task = compile_task(document, functions, model=replace(PORTLESS, period_seconds=0.5))
    assert isinstance(task, DocumentTask), task
    key = jr.key(4)
    trial = jax.jit(task.ahead)(TaskParameters(), key, key)
    assert trial.present["seen"].tolist() == [False, True, True, True]
    assert trial.variable_defined["velocity"].tolist() == [False, True, True, True]
    np.testing.assert_array_equal(trial.inputs["seen"], [0.0, 1.0, 1.0, 1.0])


def test_nested_reset_updates_observations_and_delay_queues():
    interface, graph = _pair_model(writable_gru=True)
    field, = (field for field in interface.reset.fields if field.address.composites)
    assert field.address.composites == ("wrapper", "controller")
    assert field.address.element.path == ("reset", "hidden")
    identifier = field.address.to_text()
    key = jr.key(32)
    hidden = jnp.ones(field.type.shape, dtype=jnp.int32)
    reset = interface.reset.prepare(graph, {identifier: hidden}, key=key)
    controller = graph.nodes["wrapper"].nodes["controller"]
    cell = controller.nodes["cell"]
    state = cell.state_view(reset.state)
    assert state.hidden.dtype == jnp.dtype(field.dtype)
    np.testing.assert_array_equal(state.hidden, hidden)
    np.testing.assert_array_equal(cell.initial_outputs(state)["output"], hidden)
    np.testing.assert_array_equal(reset.outputs["hidden"], hidden)
    carry = reset.cycle_init["__nested__", "wrapper"]["__nested__", "controller"]
    np.testing.assert_array_equal(carry["readout", "input"], jnp.stack((hidden, hidden)))
    default = interface.reset.prepare(graph, {}, key=key)
    np.testing.assert_array_equal(cell.state_view(default.state).hidden, 0)
    renamed, _ = _pair_model(writable_gru=True, rename=True)
    assert renamed.reset.fields == interface.reset.fields


def test_writable_reset_fields_refuse_read_only_shapes_and_dtypes(paired_model):
    interface, graph = paired_model
    writable = reset_position(interface)
    unsupported = writable.model_copy(update={"element": writable.element.model_copy(update={
        "path": ("reset", "effector", "force"),
    })})
    assert {field.address.element.path for field in interface.reset.fields} == {
        ("reset", "effector", "position"), ("reset", "effector", "velocity"),
    }
    key = jr.key(0)
    for address, value, message in (
        (unsupported, [0.0, 0.0], "not a declared writable"),
        (writable, [1.0], "shape"),
        (writable, [True, False], "real"),
    ):
        with pytest.raises(ModelResetError, match=message) as error:
            interface.reset.prepare(graph, {address.to_text(): value}, key=key)
        assert error.value.address == address.to_text()


def test_trial_configuration_changes_force_without_mutating_model():
    interface, graph = _pair_model(curl=True)
    fields = {field.address.element.path: field for field in interface.reset.fields}
    assert {path for path in fields if path[0] == "fields"} == {
        ("fields", "active"), ("fields", "scale"), ("fields", "amplitude"),
    }
    key = jr.key(11)
    updates = {
        ("reset", "effector", "position"): [0.5, -0.25],
        ("reset", "effector", "velocity"): [2.0, 1.0],
        ("fields", "active"): True,
        ("fields", "scale"): 2,
        ("fields", "amplitude"): 3,
    }
    inits = {fields[path].address.to_text(): value for path, value in updates.items()}
    reset = interface.reset.prepare(graph, inits, key=key)
    plant = graph.nodes["plant"]
    curl = graph.nodes["curl"]
    effector = plant.state_view(reset.state).effector
    np.testing.assert_array_equal(effector.pos, updates["reset", "effector", "position"])
    np.testing.assert_array_equal(effector.vel, updates["reset", "effector", "velocity"])
    inputs = {"effector": effector, "force": jnp.zeros((2,), dtype=jnp.float32)}
    active, _ = curl(inputs, reset.state, key=key)
    default = interface.reset.prepare(graph, {}, key=key)
    inactive, _ = curl(inputs, default.state, key=key)
    np.testing.assert_array_equal(active["force"], [-6, 12])
    np.testing.assert_array_equal(inactive["force"], 0)
    assert not bool(curl.configuration_values()["active"])


@pytest.mark.parametrize("kind", ["simple", "delayed", "stabilization"])
def test_document_pair_drives_signed_model_graph_with_real_reset(kind, functions, paired_model):
    interface, graph = paired_model
    old = legacy(kind)
    doc, task = example(kind, functions, interface=interface, old=old)
    (port,) = interface.inputs
    (command,) = interface.outputs
    assert port.required and port.type == POINT
    assert ModelAddress.from_text(port.id).element.end.port == "task-input"
    assert task.period == graph.period_seconds == graph.nodes["plant"].dt == PERIOD
    refused = compile_task(doc, functions, model=interface)
    assert any(problem.path == ("model", "inputs", port.id) for problem in refused)
    bad = compile_task(
        doc,
        functions,
        model=interface,
        input_overrides={"reach_target": "target_vel"},
    )
    assert bad[0].code.value == "type_mismatch"
    read_only = reset_position(interface)
    read_only = read_only.model_copy(update={"element": read_only.element.model_copy(update={
        "path": ("fields", "mass"),
    })})
    bad = compile_task(
        doc.model_copy(update={"links": (Link(model_field=read_only.to_text(), variable="start"),)}),
        functions, model=interface, input_overrides={"reach_target": "target_pos"},
    )
    assert bad[0].path == ("links", read_only.to_text())
    assert bad[0].code.value == "unknown_name"

    run_key = jr.key(8)
    keys = jr.split(run_key, 6)
    trials = jax.jit(jax.vmap(lambda key: task.ahead(TaskParameters(), run_key, key)))(keys)
    trace = (
        GraphTraceRequest("state_path", "hand", path="plant.effector.pos"),
        GraphTraceRequest("state_path", "velocity", path="plant.effector.vel"),
    )

    def run(trial, key):
        inputs = task.model_inputs(trial)
        prepared = interface.reset.prepare(graph, trial.inits, key=key, inputs=inputs)
        before = graph.nodes["plant"].state_view(prepared.state).effector
        outputs, _, retained = graph.rollout_with_trace(
            inputs,
            prepared.state,
            key=key,
            n_steps=task.n_steps,
            trace=trace,
            cycle_init=prepared.cycle_init,
        )
        return before, outputs["command"], retained

    initial, output, retained = jax.jit(jax.vmap(run))(trials, keys)
    np.testing.assert_array_equal(initial.pos, trials.inits[reset_position(interface).to_text()])
    np.testing.assert_array_equal(initial.vel, 0)
    np.testing.assert_array_equal(initial.force, 0)
    assert output.shape == (6, STEPS, 2)
    np.testing.assert_allclose(
        retained["hand"][:, 0],
        initial.pos + retained["velocity"][:, 0] * (PERIOD / 2),
        atol=2e-6,
        rtol=0,
    )
    np.testing.assert_array_equal(
        task.model_inputs(trials)["reach_target"], trials.inputs["target_pos"]
    )
    term = ObjectiveTerm(
        id="effort",
        measure=expression("sqnorm(command)", (command,)),
        reduction=Reduction.SUM,
        epochs=("move",),
    )
    executor = compile_objective(
        ObjectiveDocument(id="effort", terms=(term,)),
        (command,),
        functions,
        task=task,
    )
    assert isinstance(executor, ObjectiveExecutor), executor
    loss = executor.evaluate({command.id: output}, trials, run_key=run_key, trial_keys=keys)
    mask = jnp.arange(STEPS) >= trials.epochs.starts[:, -1, None]
    expected = jnp.mean(jnp.sum(jnp.where(mask, jnp.sum(output**2, axis=-1), 0), axis=-1))
    assert int(loss.terms["effort"].covered) == len(keys)
    assert float(loss.loss) == pytest.approx(float(expected), rel=2e-6)


@pytest.mark.parametrize("kind", ["simple", "stabilization"])
def test_reach_and_stabilization_sample_distributions(kind, functions):
    _, task = example(kind, functions)
    old = legacy(kind)
    before_keys = jr.split(jr.key(24), 20_000)
    after_keys = jr.split(jr.key(25), 20_000)

    def legacy_positions(key):
        trial = old.get_train_trial(key)
        return trial.inits["mechanics.effector"].pos, trial.targets["mechanics.effector.pos"].value[
            0
        ]

    def document_positions(key):
        prepared, _ = task.init(TaskParameters(), key, key)
        return prepared.values["start"], prepared.values["target"]

    before = jax.jit(jax.vmap(legacy_positions))(before_keys)
    after = jax.jit(jax.vmap(document_positions))(after_keys)
    for old_values, new_values in zip(before, after, strict=True):
        for axis in range(2):
            assert (
                ks_2samp(np.asarray(old_values[:, axis]), np.asarray(new_values[:, axis])).statistic
                < 0.025
            )
    if kind == "stabilization":
        np.testing.assert_array_equal(after[0], after[1])
