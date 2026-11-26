"""Capture and replay the exact start and compiled blocks of a Train segment."""

from __future__ import annotations

from collections.abc import Callable
from contextlib import ExitStack
from dataclasses import dataclass, field
from hashlib import sha256
from typing import TYPE_CHECKING, Literal

import equinox as eqx
import jax
import numpy as np

from feedbax.compiler.trainability import model_trainability
from feedbax.contracts.artifact_custody import (
    CONTENT_OBJECT_REF_PREFIX, ContentBinding, ContentObjectRef, ContentTree, ContentTreeFile, ContentTreeRef,
)
from feedbax.contracts.canonical_json import canonical_json_v2_bytes
from feedbax.contracts.computation_validity import (
    REFERENCE_PLATFORM, ArgumentIdentity, ArgumentSignature, ComputationDifferent, ComputationStart,
    ComputationTrace, ComputationUnconfirmable, DriverChanged, DriverIdentity, PrefixChanged, PrefixConfirmed, ProgramIdentity,
    TrainPrefix,
)
from feedbax.contracts.effective_success import RequestObject
from feedbax.contracts.method_checkpoint import CheckpointLatestPointer
from feedbax.contracts.request_refs import ReferenceValue
from feedbax.training.train_driver import TRAIN_DRIVER_REVISION, TrainBlock, TrainSchedule, drive_train
from feedbax.validity.computation import (
    UnrecordableArguments, argument_signature, capture_arguments, capture_program, capture_train_prefix,
    compare_computation, compare_environment_prefix, pytree_operands,
)

if TYPE_CHECKING:
    from feedbax.contracts.code_validity import ComputationUnavailable
    from feedbax.contracts.effective_success import Bindings
    from feedbax.contracts.receipt_validity import CheckOutcome, ValidityCheckDelivery, ValidityCheckEvidence
    from feedbax.persistence.artifact_custody import ImmutableArtifactBlobProvider
    from feedbax.application.bundle import ApplicationRegistryBundle
    from feedbax.runtime.graph import Graph
    from feedbax.training.method_runtime import BlockResult, LoweredBlock, TrainingLoop
    from feedbax.training.methods import TrainingState
    from feedbax.host.training_worker import _PreparedTrain


_SLOT = "updates"
_CHECKPOINT_EVIDENCE = "receipt.json"
TRAIN_DRIVER = DriverIdentity(name="train", revision=TRAIN_DRIVER_REVISION)


class CheckpointStart(ReferenceValue):
    kind: Literal["predecessor", "resume"]
    pointer: CheckpointLatestPointer
    content: ContentBinding
    source_request: RequestObject | None = None


class TrainReplay(ReferenceValue):
    request: RequestObject
    checkpoint_catalog: tuple[int, ...]
    checkpoint: CheckpointStart | None = None

    @property
    def contents(self) -> tuple[ContentObjectRef | ContentTreeRef, ...]:
        refs = (self.request.object, *self.request.documents)
        if self.checkpoint is None:
            return refs
        source = self.checkpoint.source_request
        return (*refs, self.checkpoint.content.tree, *((source.object, *source.documents) if source else ()))


def _operands(state: TrainingState, operands: tuple[jax.Array, ...]):
    return pytree_operands((state, *operands))


def capture_train_start(loop: TrainingLoop, state: TrainingState, block: TrainBlock) -> ArgumentIdentity:
    arrays, static = _operands(state, loop.block_arguments(block.completed, block.count))
    return capture_arguments(arrays, static=static)


def capture_train_program(
    state: TrainingState, operands: tuple[jax.Array, ...], count: int, lowered: LoweredBlock, *,
    execution_platform: str | None = None,
) -> ProgramIdentity:
    arrays, static = _operands(state, operands)
    signature = argument_signature(arrays, static=[count, static])
    platform = execution_platform or lowered.platform
    execution = lowered.execution if platform == lowered.platform else lowered.traced.lower(lowering_platforms=(platform,))
    reference = None
    if platform != REFERENCE_PLATFORM:
        try:
            reference = (lowered.execution if lowered.platform == REFERENCE_PLATFORM
                         else lowered.traced.lower(lowering_platforms=(REFERENCE_PLATFORM,)))
        except NotImplementedError:
            # Some accelerator-only primitives have no reference lowering.
            pass
    return capture_program(
        execution, slot=_SLOT, specialization=signature, execution_platform=platform, reference_lowered=reference,
    )


@dataclass
class TrainOutputPrefix:
    end: int
    losses: list[np.ndarray] = field(default_factory=list)
    model: Graph | None = None
    saved: TrainPrefix | None = None

    def record(self, block: TrainBlock, result: BlockResult) -> None:
        if block.completed >= self.end:
            return
        losses = result.metrics.get("train_loss")
        if not isinstance(losses, np.ndarray) or losses.ndim != 2 or losses.shape[1] != 1:
            return
        taken = min(losses.shape[0], self.end - block.completed)
        if taken:
            self.losses.append(losses[:taken, 0])
            self.model = result.state.model if result.completed < self.end else result.sampled_model
            self.saved = None

    def retain(self, store_blob: Callable[[bytes], ContentObjectRef]) -> TrainPrefix | None:
        from feedbax.training.method_runtime import replica_value

        if not self.losses or self.model is None:
            return None
        if self.saved is None:
            model = replica_value(self.model, 0)
            _, trainable = model_trainability(model)
            leaves, _ = jax.tree.flatten_with_path(eqx.filter(model, trainable))
            parameters = {jax.tree_util.keystr(path): array for path, array in leaves}
            self.saved = capture_train_prefix(np.concatenate(self.losses), parameters, store_blob=store_blob)
        return self.saved


class TrainCapture:
    """Gather actual launched block specializations and retain their continuation recipe."""

    def __init__(
        self, loop: TrainingLoop, state: TrainingState, schedule: TrainSchedule, replay: TrainReplay,
        store: ImmutableArtifactBlobProvider,
    ) -> None:
        self.schedule = schedule
        self.replay = replay
        self.store = store
        self.prefix = TrainOutputPrefix(schedule.prefix_end)
        self.programs: dict[ArgumentSignature, ProgramIdentity] = {}
        self.unavailable: str | None = None
        self.start: ArgumentIdentity | None
        first = next(schedule.blocks(), None)
        try:
            self.start = capture_train_start(loop, state, first) if first is not None else None
        except UnrecordableArguments as error:
            self.start = None
            self.unavailable = str(error)

    def observe(self, state: TrainingState, operands: tuple[jax.Array, ...], count: int, lowered: LoweredBlock) -> None:
        if self.start is not None:
            try:
                program = capture_train_program(state, operands, count, lowered)
            except UnrecordableArguments as error:
                self.start = None
                self.unavailable = str(error)
                return
            self.programs[program.specialization] = program

    def snapshot(self, completed: int) -> ComputationTrace | ComputationUnavailable | None:
        steps = completed - self.schedule.start
        if steps == 0:
            return None
        if self.unavailable is not None:
            from feedbax.contracts.code_validity import ComputationUnavailable

            return ComputationUnavailable(
                reason="arguments", driver=TRAIN_DRIVER, completed_steps=steps, detail=self.unavailable,
            )
        if self.start is None or not self.programs:
            raise ValueError("Train committed updates before recording their lowered computation")
        binding = self.store.store_object(canonical_json_v2_bytes(self.replay.model_dump(mode="json")))
        return ComputationTrace(
            driver=TRAIN_DRIVER,
            starts=(ComputationStart(
                slot=_SLOT, arguments=self.start, binding=binding, binding_inputs=self.replay.contents,
            ),),
            programs=tuple(self.programs.values()),
            completed_steps=steps,
            prefix=self.prefix.retain(self.store.store_object),
        )


def archive_checkpoint_validity(store: ImmutableArtifactBlobProvider) -> ContentTreeRef | None:
    """Encode current immutable evidence for the checkpoint's Store closure."""
    from feedbax.persistence.evidence import evidence_objects
    from feedbax.validity.recording import snapshot_execution

    evidence = snapshot_execution(store)
    if evidence is None:
        return None
    record = store.store_object(canonical_json_v2_bytes(evidence.model_dump(mode="json")))
    objects = {ref.sha256: ref for ref in evidence_objects(store, evidence.trace)}
    return store.store_tree(ContentTree(
        directories=("objects",) if objects else (),
        files=(
            *(ContentTreeFile(relative_path=f"objects/{key}", object=ref) for key, ref in sorted(objects.items())),
            ContentTreeFile(relative_path=_CHECKPOINT_EVIDENCE, object=record),
        ),
    ))


def checkpoint_evidence(store: ImmutableArtifactBlobProvider, tree: ContentTreeRef) -> ContentObjectRef:
    (record,) = (file.object for file in store.get_tree(tree).files if file.relative_path == _CHECKPOINT_EVIDENCE)
    return record


def _reprepare(
    store: ImmutableArtifactBlobProvider, replay: TrainReplay, registries: ApplicationRegistryBundle,
    views: ExitStack, *, owner: str, inputs: Bindings,
) -> _PreparedTrain:
    from feedbax.contracts.effective_success import read_success_request
    from feedbax.contracts.request_specs import DocumentTrainingRequestSpec
    from feedbax.contracts.studio_training import PublishedTrainingCheckpoint
    from feedbax.contracts.train_settings import normalize_train_method
    from feedbax.persistence.method_checkpoint import materialize_checkpoint_content, verify_method_checkpoint_content
    from feedbax.persistence.canonical_train import (
        BoundStudioTrainingJobInput, CanonicalTrainRequestBinding, _decode_train, resolve_training_request_binding,
    )
    from feedbax.host.training_worker import _prepare_train

    request = replay.request
    documents = {ref.sha256: ref for ref in request.documents}
    spec, checkpoint, imported_model, model = _decode_train(
        request.id, store.get_object(request.object), lambda digest: store.get_object(documents[digest]),
    )
    model_receipt = None
    if model is not None:
        (source,) = inputs["checkpoint"]
        if source.selection is None:
            raise ValueError("Train receipt does not pin its synthesized starting model.")
        model_receipt = source.selection.receipt
    spec, _, _ = normalize_train_method(spec, registries.training_methods)
    launch = BoundStudioTrainingJobInput(
        spec=spec, project_id="validity-check", request_id=request.id, manifest_root=store.root,
        canonical_request=CanonicalTrainRequestBinding(
            request=request.object, documents=request.documents, checkpoint=checkpoint, imported_model=imported_model,
            model=model, model_receipt=model_receipt,
        ), checkpoint_catalog=replay.checkpoint_catalog,
    )
    binding = resolve_training_request_binding(launch, registries=registries)
    producer = source_spec = None
    resumed = False
    if (saved := replay.checkpoint) is not None:
        verify_method_checkpoint_content(store, saved.pointer, saved.content)
        source = views.enter_context(materialize_checkpoint_content(saved.content, store, owner=owner))
        producer = PublishedTrainingCheckpoint(
            checkpoint_root=source, completed_updates=saved.pointer.completed_updates, pointer=saved.pointer,
        )
        resumed = saved.kind == "resume"
        if saved.source_request is not None:
            source_spec = read_success_request(saved.source_request, store.get_object).spec
            if not isinstance(source_spec, DocumentTrainingRequestSpec):
                raise ValueError("Starting checkpoint source is not a Train request")
    return _prepare_train(
        launch, binding=binding, registries=registries, producer=producer,
        resumed=resumed, source_spec=source_spec, phase=lambda _: None,
    )


def _check_code(
    previous: ComputationTrace, present: ComputationTrace, prepared: _PreparedTrain,
    store: ImmutableArtifactBlobProvider,
) -> CheckOutcome:
    from feedbax.contracts.receipt_validity import CheckOutcome

    if previous.driver != present.driver:
        return CheckOutcome(
            outcome="different", changes=(DriverChanged(previous=previous.driver, present=present.driver),),
        )
    program_platforms = {program.execution.platform for program in previous.programs}
    if len(program_platforms) != 1:
        return CheckOutcome(outcome="unavailable", detail="Train segment has inconsistent execution targets")
    (platform,) = program_platforms
    programs: dict[ArgumentSignature, ProgramIdentity] = {}
    counts: set[int] = set()
    end = prepared.schedule.start + previous.completed_steps
    for block in prepared.schedule.blocks():
        if block.completed >= end:
            break
        if block.count in counts:
            continue
        counts.add(block.count)
        lowered = prepared.loop.lower_block(prepared.state, completed=block.completed, count=block.count)
        program = capture_train_program(
            prepared.state, prepared.loop.block_arguments(block.completed, block.count), block.count, lowered,
            execution_platform=platform,
        )
        programs[program.specialization] = program
    if not programs or end > prepared.schedule.target:
        return CheckOutcome(outcome="different", detail="The recorded updates no longer fit the prepared Train clock")
    present = present.model_copy(update={"programs": tuple(programs.values())})
    comparison = compare_computation(previous, present, read_blob=store.get_object)
    if isinstance(comparison, ComputationUnconfirmable):
        return CheckOutcome(outcome="unavailable", detail=comparison.reason)
    if isinstance(comparison, ComputationDifferent):
        return CheckOutcome(outcome="different", changes=comparison.changes)
    return CheckOutcome(outcome="equal")


def _check_environment(
    previous: ComputationTrace, present: ComputationTrace, prepared: _PreparedTrain,
    store: ImmutableArtifactBlobProvider,
) -> CheckOutcome:
    from feedbax.contracts.receipt_validity import CheckOutcome

    if present.starts[0].arguments != previous.starts[0].arguments:
        return CheckOutcome(outcome="different", change="arguments")
    if not isinstance(previous.prefix, TrainPrefix) or len(previous.prefix.losses) != 5:
        return CheckOutcome(outcome="unavailable", detail="The qualified five-update Train witness is unavailable")

    witness = TrainOutputPrefix(prepared.schedule.prefix_end)
    outputs: dict[str, bytes] = {}

    def keep(raw: bytes) -> ContentObjectRef:
        digest = sha256(raw).hexdigest()
        ref = ContentObjectRef(object_id=CONTENT_OBJECT_REF_PREFIX + digest, sha256=digest, size_bytes=len(raw))
        outputs[digest] = raw
        return ref

    evaluated = drive_train(
        prepared.schedule, prepared.state,
        step=lambda state, block: prepared.loop.run_block(
            state, completed=block.completed, count=block.count, limit=block.limit,
        ),
        record=witness.record, checkpoint=lambda _: None, should_stop=lambda: False,
        end=prepared.schedule.prefix_end,
    )
    if evaluated.rejected:
        return CheckOutcome(outcome="different", change="rejected_update")
    present = present.model_copy(update={"prefix": witness.retain(keep)})
    measured = compare_environment_prefix(
        previous, present, read_blob=lambda ref: outputs[ref.sha256] if ref.sha256 in outputs else store.get_object(ref),
    )
    if isinstance(measured, PrefixConfirmed):
        return CheckOutcome(outcome="equal", drift=measured.drift)
    if isinstance(measured, PrefixChanged):
        return CheckOutcome(outcome="different", change=measured.reason, drift=measured.drift)
    return CheckOutcome(outcome="unavailable", detail=measured.reason)


def recheck_train(delivery: ValidityCheckDelivery, registries: ApplicationRegistryBundle) -> ValidityCheckEvidence:
    """Read the exact receipt and inspect every contributing Train segment."""
    from feedbax.contracts.receipt_validity import CheckOutcome, ValidityCheckEvidence
    from feedbax.persistence.artifact_custody import ImmutableArtifactBlobProvider
    from feedbax.persistence.effective_success import SuccessView
    from feedbax.persistence.evidence import read_execution_evidence

    store = ImmutableArtifactBlobProvider(delivery.store_root)
    receipt = SuccessView(store).read_receipt(delivery.receipt)
    if receipt.request.id != delivery.request_id:
        raise ValueError("Train recheck receipt differs from the selected request")

    def outcomes():
        trace = receipt.trace
        owner = receipt.attempt
        while trace is not None:
            previous = trace.computation
            if previous is None or len(previous.starts) != 1 or previous.starts[0].binding is None:
                unavailable = CheckOutcome(outcome="unavailable", detail=f"Train segment {owner} has no exact replay start")
                yield {domain: unavailable for domain in delivery.checks}
            else:
                start = previous.starts[0]
                assert start.binding is not None
                replay = TrainReplay.model_validate_json(store.get_object(start.binding))
                if replay.request != receipt.request or start.binding_inputs != replay.contents:
                    raise ValueError("Train recheck recipe differs from its retained request and input closure")
                with ExitStack() as views:
                    prepared = _reprepare(store, replay, registries, views, owner=owner, inputs=receipt.inputs)
                    arguments = capture_train_start(prepared.loop, prepared.state, next(prepared.schedule.blocks()))
                    present = previous.model_copy(update={
                        "driver": TRAIN_DRIVER, "starts": (start.model_copy(update={"arguments": arguments}),),
                        "prefix": None,
                    })
                    observed = {}
                    for domain in delivery.checks:
                        try:
                            result = (_check_code(previous, present, prepared, store) if domain == "code" else
                                      _check_environment(previous, present, prepared, store))
                        except (NotImplementedError, UnrecordableArguments) as error:
                            result = CheckOutcome(outcome="unavailable", detail=str(error) or type(error).__name__)
                        observed[domain] = result
                    yield observed
            if not trace.prefixes:
                break
            (ref,) = trace.prefixes
            inherited = read_execution_evidence(store, ref)
            trace, owner = inherited.trace, inherited.attempt

    by_domain = {domain: [] for domain in delivery.checks}
    for segment in outcomes():
        for domain, result in segment.items():
            by_domain[domain].append(result)
    priority = {"equal": 0, "unavailable": 1, "different": 2}
    return ValidityCheckEvidence(**{
        domain: max(results, key=lambda result: priority[result.outcome]) if results else
        CheckOutcome(outcome="unavailable", detail="Train receipt has no execution evidence")
        for domain, results in by_domain.items()
    })
