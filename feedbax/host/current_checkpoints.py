"""Read the current scientific value of authenticated checkpoint selections."""

from __future__ import annotations

from dataclasses import dataclass
import json
from sqlite3 import Row
from typing import TYPE_CHECKING

from feedbax.contracts.effective_success import BoundMember, EvaluationSubjectValue, Product, Success
from feedbax.contracts.jobs import JobReceipt
from feedbax.contracts.manifest import training_run_manifest_id
from feedbax.contracts.receipt_validity import InputDifference
from feedbax.contracts.request_refs import BestRank, ContentRef, RequestId
from feedbax.contracts.rollout import ResolvedRolloutMember
from feedbax.contracts.studio_training import PublishedTrainingCheckpoint
from feedbax.host.best_selection import completion_score, rank_train_metric, ranked_requests
from feedbax.host.checkpoint_selection import CheckpointSelection, is_checkpoint_producer, requested_checkpoint_selection
from feedbax.persistence.checkpoint_publication import model_product, read_retained_checkpoint
from feedbax.persistence.effective_success import SuccessView
from feedbax.persistence.method_checkpoint import read_method_checkpoint_manifest, verify_method_checkpoint_content
from feedbax.persistence.receipt_content import verify_success_content
from feedbax.persistence.store_damage import inspect_without_recording
from feedbax.persistence.train_job import read_train_job

if TYPE_CHECKING:
    from feedbax.host.store import Store


@dataclass(frozen=True)
class _Producer:
    attempt: str
    anchor: str | None
    tip: str | None
    success: Success | None
    path: tuple[str, ...]


class CurrentCheckpoints:
    """One project Store's current producer and checkpoint observations."""

    def __init__(self, host: Store, view: SuccessView, project: str):
        self.host = host
        self.view = view
        self.project = project
        self.root = str(view.store.root)
        self.producers: dict[RequestId, _Producer | None] = {}
        self.native: dict[tuple[str, str | None, CheckpointSelection | None], dict | None] = {}

    def _belongs(self, row: Row, request: RequestId) -> bool:
        return (
            (row["project"], row["request"]) == (self.project, request)
            and json.loads(row["data"]).get("manifest_root") == self.root
        )

    def _native_anchor(self, row: Row | None, request: RequestId) -> tuple[str | None, tuple[str, ...]]:
        path = []
        while row is not None and self._belongs(row, request):
            path.append(row["id"])
            if row["continues"] is None:
                launch, data = json.loads(row["launch"]), json.loads(row["data"])
                if is_checkpoint_producer(launch, continues=None, manifest_root=self.root,
                                          collected=data.get("collected_train", False)):
                    return row["id"], tuple(path)
                break
            row = self.host.get(row["continues"])
        return None, tuple(path)

    def _producer(self, request: RequestId) -> _Producer | None:
        if request not in self.producers:
            self.producers[request] = self._read_producer(request)
        return self.producers[request]

    def _read_producer(self, request: RequestId) -> _Producer | None:
        if (ref := self.view.authored_ref(request)) is not None:
            success = self.view.read_receipt(ref)
            if success.request.id != request:
                raise _SelectionUnavailable("the authored Train receipt names a different request")
            row = self.host.get(success.attempt)
            anchor, path = self._native_anchor(row, request) if row is not None else (None, ())
            return _Producer(success.attempt, anchor, success.attempt, success, path or (success.attempt,))

        native, collected = [], []
        for row in self.host.db.execute(
            "SELECT * FROM jobs WHERE project=? AND request=? ORDER BY seq DESC",
            (self.project, request),
        ):
            if not self._belongs(row, request):
                continue
            data = json.loads(row["data"])
            if data.get("collected_train") and row["state"] == "stopped":
                collected.append(row["id"])
            elif row["continues"] is None and is_checkpoint_producer(
                json.loads(row["launch"]), continues=None, manifest_root=self.root,
            ):
                native.append(row["id"])
        if len(native) == 1:
            return _Producer(native[0], native[0], None, None, (native[0],))
        if not native and len(collected) == 1:
            return _Producer(collected[0], None, None, None, (collected[0],))
        return None

    def native_anchor(self, request: RequestId) -> str | None:
        try:
            producer = self._producer(request)
        except (KeyError, OSError, ValueError):
            return None
        return producer.anchor if producer is not None else None

    def _selection(self, producer: _Producer, selection: CheckpointSelection | None) -> dict | None:
        if producer.anchor is None:
            return None
        key = producer.anchor, producer.tip, selection
        if key not in self.native:
            self.native[key] = self.host._completion_candidate(
                producer.anchor, selection, tip=producer.tip, observe=selection is not None,
            ).binding
        return self.native[key]

    @inspect_without_recording()
    def compare(self, receipt: Success, launch: dict) -> InputDifference | None:
        for role, members in receipt.inputs.items():
            for bound in members:
                source = bound.checkpoint_model or bound.evaluation_subject
                if source is None:
                    continue
                try:
                    selection = requested_checkpoint_selection(receipt.request.id, launch)
                    checkpoint = (launch["payload"].get("canonical_request") or {}).get("checkpoint")
                    if isinstance(checkpoint, dict) and checkpoint.get("kind") == "best_rank":
                        current = self._best(BestRank.model_validate(checkpoint), bound)
                    elif selection is not None:
                        request = source.producer.id
                        current = request, self._models(request, selection, bound)
                    elif bound.checkpoint_model is not None or source.terminal_receipt is None:
                        raise _SelectionUnavailable("the receipt has no verifiable selection rule")
                    else:
                        return None  # Terminal subjects are compared as ordinary current products.
                except (KeyError, OSError, ValueError) as error:
                    return _input_change(role, bound, f"Current checkpoint selection is unavailable: {error}")
                return self._compare_member(role, bound, current)
        return None

    @staticmethod
    def _compare_member(
        role: str, bound: BoundMember, current: tuple[RequestId, tuple[tuple[int, Product], ...]],
    ) -> InputDifference | None:
        request, members = current
        if bound.checkpoint_model is not None:
            (_, model), = members
            if (model.content, model.descriptor) == (bound.content, bound.descriptor):
                return None
            return _input_change(role, bound, "The selected Train checkpoint model changed.", after=model.content)

        subject = bound.evaluation_subject
        assert subject is not None
        previous = EvaluationSubjectValue.model_validate(bound.descriptor)
        before = tuple(
            (member.coordinate, member.model_tree)
            for member in (previous.resolved.members if previous.resolved is not None else ())
            if isinstance(member, ResolvedRolloutMember)
        )
        after = tuple((coordinate, model.content.sha256) for coordinate, model in members)
        if (request, after) == (subject.producer.id, before):
            return None
        return _input_change(role, bound, "The selected Train subject, checkpoint coverage, or model changed.")

    def _models(
        self, request: RequestId, selection: CheckpointSelection, bound: BoundMember,
    ) -> tuple[tuple[int, Product], ...]:
        producer = self._producer(request)
        if producer is None:
            raise _SelectionUnavailable("there is no single current Train producer")
        if (binding := self._selection(producer, selection)) is not None:
            return tuple(_native_model(member) for member in binding["checkpoint_selection"]["members"])
        if producer.anchor is not None:
            raise _SelectionUnavailable("the current Train lineage has not authenticated the requested checkpoints")

        portable = self._portable_models(producer, bound)
        if all(update in portable for update in selection.coordinates):
            return tuple((update, portable[update]) for update in selection.coordinates)
        subject = bound.evaluation_subject
        if selection.allows_gaps and subject is not None and subject.closure is not None:
            terminal = subject.closure.checkpoint.occurrence.run_id
            if terminal == producer.attempt:
                value = EvaluationSubjectValue.model_validate(bound.descriptor)
                gaps = {gap.update for gap in value.gaps}
                if set(selection.coordinates) <= (set(portable) | gaps):
                    return tuple((update, portable[update]) for update in selection.coordinates if update in portable)
        raise _SelectionUnavailable("the collected Train result lacks the requested checkpoint evidence")

    def _portable_models(self, producer: _Producer, bound: BoundMember) -> dict[int, Product]:
        models = {}
        if producer.success is not None and producer.success.continuation is not None:
            (terminal,) = producer.success.products["checkpoint"]
            verify_success_content(self.view.store, terminal)
            models[producer.success.continuation.clock.completed_updates] = terminal
        elif producer.success is None:
            recovery = read_train_job(self.view.store, producer.attempt).recovery
            manifest = verify_method_checkpoint_content(
                self.view.store, recovery.pointer, recovery.content,
            )
            models[manifest.clock.completed_updates] = model_product(manifest)

        if (evidence := bound.checkpoint_model) is not None:
            candidates = ((evidence.producer, evidence.publication),)
        elif (subject := bound.evaluation_subject) is not None:
            candidates = tuple((subject.producer, source) for source in subject.checkpoints)
        else:
            candidates = ()
        for request, source in candidates:
            if source.occurrence.run_id in producer.path:
                manifest, _ = read_retained_checkpoint(self.view.store, source, request)
                product = model_product(manifest)
                verify_success_content(self.view.store, product)
                models.setdefault(manifest.clock.completed_updates, product)
        return models

    def _best(self, rule: BestRank, bound: BoundMember) -> tuple[RequestId, tuple[tuple[int, Product], ...]]:
        subject = bound.evaluation_subject
        assert subject is not None
        scores = {}
        models = {}
        retained = {item.attempt: item for item in subject.ranked_candidates}
        for candidate in rule.candidates:
            request = candidate.reference.request_id
            producer = self._producer(request)
            if producer is None:
                raise _SelectionUnavailable(f"{request} has no single current Train producer")
            if (binding := self._selection(producer, None)) is not None:
                completion = binding["completion"]
                scores[request] = completion_score(completion, rule.metric)
                models[request] = _native_model(completion)
                continue
            claim = retained.get(producer.attempt)
            if claim is None or claim.producer.id != request:
                raise _SelectionUnavailable(f"{request} lacks current authenticated evidence for {rule.metric!r}")
            ref = claim.receipt
            scores[request] = rank_train_metric(
                self.view.store.get_object(ref),
                JobReceipt(manifest_id=training_run_manifest_id(claim.attempt), manifest_sha256=ref.sha256),
                claim.attempt, rule.metric,
            )
            manifest, _ = read_retained_checkpoint(self.view.store, claim.checkpoint, claim.producer)
            product = model_product(manifest)
            verify_success_content(self.view.store, product)
            models[request] = manifest.clock.completed_updates, product
        _, winner = ranked_requests(rule, scores)
        return winner, (models[winner],)


class _SelectionUnavailable(ValueError):
    pass


def _native_model(publication: dict) -> tuple[int, Product]:
    checkpoint = PublishedTrainingCheckpoint.model_validate(publication["checkpoint"])
    manifest = read_method_checkpoint_manifest(checkpoint.checkpoint_root, checkpoint.pointer)
    return checkpoint.completed_updates, model_product(manifest)


def _input_change(role: str, bound: BoundMember, detail: str, *, after: ContentRef | None = None) -> InputDifference:
    return InputDifference(role=role, member=bound.key, before=bound.content, after=after, detail=detail)
