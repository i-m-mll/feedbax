"""Resolve Selectors on the trial's post-step sample axis."""

from dataclasses import dataclass
from typing import NamedTuple

import jax
import jax.numpy as jnp

from feedbax.contracts.expression_check import references
from feedbax.contracts.expression_registry import FunctionRegistry
from feedbax.contracts.expression_values import (
    TIME, Level, NodePath, Problem, ProblemCode, ScopedValue, Site, ValueBase, ValueType,
)
from feedbax.contracts.time_units import TimeValue, time_steps
from feedbax.contracts.window import (
    Window, WindowEpoch, WindowEpochAnchor, WindowInterval, WindowMoment, WindowOffset,
    WindowPosition, WindowRelativePosition,
)
from feedbax.expression_compile import Compiled, Inputs, compile_expression


class EpochRecord(NamedTuple):
    starts: jax.Array
    ends: jax.Array
    outcomes: jax.Array


def in_epoch(steps, start, end):
    """An unreached epoch has sentinel bounds and contributes no steps."""
    return (start >= 0) & (start <= steps) & (steps < end)


@dataclass(frozen=True)
class _Boundary:
    indices: tuple[int, ...] | None
    edge: str
    offset: int | Compiled

    def resolve(self, record, n_steps, inputs, period, *, moment=False):
        if self.indices is None:
            base = jnp.asarray([0 if self.edge == "start" else n_steps])
            valid = jnp.asarray([True])
        else:
            indices = jnp.asarray(self.indices, dtype=int)
            start, end = (jnp.take(bound, indices, axis=-1) for bound in record[:2])
            base = start if self.edge == "start" else end
            valid = (start >= 0) & ((end > start) if moment else True)
        if moment and self.edge == "end":
            base = base - 1
        steps, defined = _shift(self.offset, inputs, period)
        return base + steps, valid & defined


def _shift(offset: int | Compiled, inputs: Inputs, period: float):
    if isinstance(offset, int):
        return offset, True
    inputs = inputs._replace(parameters=offset.parameters)
    state = offset.with_trial(inputs, offset.evaluate_run(inputs))
    result, _ = offset.evaluate_step(inputs, state)
    steps = jnp.rint(result.value / period)
    return steps, result.defined & jnp.isfinite(steps)


@dataclass(frozen=True)
class _Interval:
    start: _Boundary
    end: _Boundary | int | Compiled


@dataclass(frozen=True)
class _Moment:
    at: _Boundary


@dataclass(frozen=True)
class CompiledWindow:
    """Epoch positions and expression operands resolved against one Task."""

    spans: tuple[tuple[int, ...] | _Interval | _Moment, ...]
    period: float
    references: frozenset[str]

    def reads_runtime_epoch(self, first_until: int) -> bool:
        """Whether an ahead-of-rollout mask needs a boundary known only after Until."""
        def uncertain(boundary: _Boundary, *, sample=False):
            if boundary.indices is None:
                return False
            last = max(boundary.indices, default=-1)
            return last > first_until or last == first_until and (sample or boundary.edge == "end")

        for span in self.spans:
            if isinstance(span, tuple):
                if max(span, default=-1) >= first_until:
                    return True
            elif isinstance(span, _Moment):
                if uncertain(span.at, sample=True):
                    return True
            elif uncertain(span.start) or isinstance(span.end, _Boundary) and uncertain(span.end):
                return True
        return False

    def mask(self, record: EpochRecord, n_steps: int, inputs: Inputs) -> jax.Array:
        steps = jnp.arange(n_steps)
        selected = jnp.zeros((*record.starts.shape[:-1], n_steps), dtype=bool)
        for span in self.spans:
            if isinstance(span, tuple):
                indices = jnp.asarray(span, dtype=int)
                start, end = (jnp.take(bound, indices, axis=-1)[..., None] for bound in record[:2])
                active = in_epoch(steps, start, end)
            elif isinstance(span, _Interval):
                start, valid = span.start.resolve(record, n_steps, inputs, self.period)
                if isinstance(span.end, _Boundary):
                    end, reached = span.end.resolve(record, n_steps, inputs, self.period)
                else:
                    shift, reached = _shift(span.end, inputs, self.period)
                    end = start + shift
                active = (valid & reached)[..., None] & (start[..., None] <= steps) & (steps < end[..., None])
            else:
                at, valid = span.at.resolve(record, n_steps, inputs, self.period, moment=True)
                active = valid[..., None] & (at[..., None] == steps)
            selected |= jnp.any(active, axis=-2)
        return selected


def compile_window(
    window: Window,
    *,
    epochs: tuple[str, ...],
    period: float,
    names: tuple[ScopedValue, ...],
    functions: FunctionRegistry,
    field_address: tuple[str, ...],
    path: NodePath,
    nominal: bool = False,
) -> CompiledWindow | tuple[Problem, ...]:
    """Bind anchors by identity and pair each constant offset with the Model clock."""
    problems: list[Problem] = []
    used: set[str] = set()
    spans = []
    site = Site(
        expected=ValueType(base=ValueBase.REAL, dimensions=TIME), finest=Level.TRIAL, names=names,
    )

    def occurrences(epoch: WindowEpoch, at: NodePath) -> tuple[int, ...]:
        found = tuple(index for index, id in enumerate(epochs) if id == epoch.epoch)
        if not found:
            problems.append(Problem(
                code=ProblemCode.UNKNOWN_NAME, path=(*at, "epoch"), reason=f"Unknown epoch {epoch.epoch!r}.",
            ))
            return ()
        if epoch.occurrence is None:
            return found
        if epoch.occurrence >= len(found):
            problems.append(Problem(
                code=ProblemCode.OUT_OF_SCOPE, path=(*at, "occurrence"),
                reason=f"Epoch {epoch.epoch!r} has no occurrence {epoch.occurrence}.",
            ))
        return found[epoch.occurrence:epoch.occurrence + 1]

    def offset(value: WindowOffset | None, at: NodePath) -> int | Compiled:
        if value is None:
            return 0
        if isinstance(value, TimeValue):
            try:
                return time_steps(value, period)
            except ValueError as error:
                problems.append(Problem(code=ProblemCode.STRUCTURE, path=at, reason=str(error)))
                return 0
        compiled = compile_expression(
            value, site, functions, field_address=(*field_address, *map(str, at[len(path):])),
            period_seconds=period, path=at, nominal=nominal,
        )
        if isinstance(compiled, Compiled):
            used.update(references(compiled.checked.tree))
            return compiled
        problems.extend(compiled)
        return 0

    def boundary(value: WindowPosition, at: NodePath) -> _Boundary:
        anchor = value.anchor
        return _Boundary(
            occurrences(anchor, (*at, "anchor")) if isinstance(anchor, WindowEpochAnchor) else None,
            anchor.edge, offset(value.offset, (*at, "offset")),
        )

    for index, span in enumerate(window.spans):
        at = (*path, "spans", index)
        if isinstance(span, WindowEpoch):
            spans.append(occurrences(span, at))
        elif isinstance(span, WindowMoment):
            spans.append(_Moment(boundary(span.at, (*at, "at"))))
        else:
            assert isinstance(span, WindowInterval)
            start = boundary(span.start, (*at, "start"))
            if isinstance(span.end, WindowRelativePosition):
                end = offset(span.end.offset, (*at, "end", "offset"))
            else:
                end = boundary(span.end, (*at, "end"))
                a, b = start.indices, end.indices
                if a and b and len(a) > 1 and len(b) > 1 and len(a) != len(b):
                    problems.append(Problem(
                        code=ProblemCode.STRUCTURE, path=(*at, "end", "anchor"),
                        reason="Repeated Window boundaries must have the same number of occurrences.",
                    ))
            spans.append(_Interval(start, end))
    return tuple(problems) if problems else CompiledWindow(tuple(spans), period, frozenset(used))
