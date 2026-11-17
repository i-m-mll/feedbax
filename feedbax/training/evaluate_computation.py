"""Computation evidence for ordinary and Train-hosted Evaluate calls."""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING
from weakref import WeakKeyDictionary

from feedbax.contracts.computation_validity import (
    ComputationStart, ComputationTrace, EvaluateOutput, EvaluatePrefix, ProgramIdentity,
)
from feedbax.contracts.live_evaluate import EVALUATE_DRIVER, EVALUATE_PREFIX_STEPS, rollout_slot

if TYPE_CHECKING:
    import jax

    from feedbax.persistence.array_container import EncodedArrayContainer
    from feedbax.persistence.artifact_custody import ImmutableArtifactBlobProvider
    from feedbax.runtime.trial_execution import TrialEvaluation
    from feedbax.training.rollout_scoring import LoweredRollout


def drive_replicas[T, U](
    coordinate: int | None, replicas: tuple[tuple[str, T], ...], *, seed: int,
    call: Callable[[str, T, int], U], stopped: Callable[[], bool],
) -> tuple[U, ...] | None:
    results = []
    for replica, operand in replicas:
        if stopped():
            return None
        results.append(call(rollout_slot(coordinate, replica), operand, seed))
    return None if stopped() else tuple(results)


class RolloutExecutables:
    """Share compatible compiled programs within one delivery."""

    def __init__(self):
        self._by_lowering: WeakKeyDictionary[jax.stages.Lowered, jax.stages.Compiled] = WeakKeyDictionary()
        self._by_program: dict[tuple, jax.stages.Compiled] = {}

    def run(self, call: LoweredRollout) -> TrialEvaluation:
        import jax

        if (executable := self._by_lowering.get(call.lowered)) is None:
            program = call.program.execution
            if all(foreign.kind == "environment" for foreign in program.calls):
                key = program.sha256, call.program.specialization, call.lowered.out_tree
                if (executable := self._by_program.get(key)) is None:
                    self._by_program[key] = executable = call.lowered.compile()
            else:
                executable = call.lowered.compile()
            self._by_lowering[call.lowered] = executable
        return jax.block_until_ready(executable(*call.operands))


class EvaluationComputation:
    def __init__(self, store: ImmutableArtifactBlobProvider):
        self.store = store
        self.starts: list[ComputationStart] = []
        self.programs: list[ProgramIdentity] = []
        self.outputs: list[tuple[str, EncodedArrayContainer | EvaluateOutput]] = []

    def observe(self, call: LoweredRollout) -> None:
        self.starts.append(call.start)
        self.programs.append(call.program)

    def observe_output(self, slot: str, evaluated: TrialEvaluation) -> None:
        if len(self.outputs) == EVALUATE_PREFIX_STEPS:
            return
        import jax
        import numpy as np

        from feedbax.persistence.array_container import SUPPORTED_DTYPES, encode_array_container

        arrays, metadata = {}, []
        for index, (path, value) in enumerate(jax.tree.leaves_with_path(evaluated)):
            dtype = str(value.dtype)
            if jax.dtypes.issubdtype(value.dtype, jax.dtypes.prng_key):
                value = jax.random.key_data(value)
            array = np.asarray(value)
            metadata.append({"path": jax.tree_util.keystr(path), "dtype": dtype, "shape": list(array.shape)})
            if array.dtype.name not in SUPPORTED_DTYPES:
                array = np.asarray(array.reshape(-1), dtype=array.dtype.newbyteorder("<"), order="C").view(np.uint8)
            arrays[str(index)] = array
        encoded = encode_array_container(arrays, metadata=metadata)
        self.outputs.append((slot, encoded))

    def inherit(self, computation: ComputationTrace) -> None:
        self.starts.extend(computation.starts)
        self.programs.extend(computation.programs)
        if isinstance(computation.prefix, EvaluatePrefix):
            for output in computation.prefix.outputs[:EVALUATE_PREFIX_STEPS - len(self.outputs)]:
                self.outputs.append((output.slot, output))

    def snapshot(self) -> ComputationTrace | None:
        if not self.starts:
            return None
        outputs = []
        for start, (slot, value) in zip(self.starts, self.outputs):
            if start.slot != slot:
                break
            if not isinstance(value, EvaluateOutput):
                value = EvaluateOutput(
                    slot=slot, values=self.store.store_object(value.data),
                    descriptor_sha256=value.descriptor_sha256,
                )
            outputs.append(value)
        return ComputationTrace(
            driver=EVALUATE_DRIVER, starts=tuple(self.starts), programs=tuple(self.programs),
            completed_steps=len(self.starts), prefix=EvaluatePrefix(outputs=tuple(outputs)) if outputs else None,
        )
