"""Find authored sibling members that share a shape after abstracting expression literals."""

from collections.abc import Hashable, Mapping

from pydantic import BaseModel, ValidationError

from feedbax.contracts.expression_values import LiteralValue, UnresolvedValue
from feedbax.contracts.member_family import StoredFamilyMember
from feedbax.contracts.objective_document import STORED_OBJECTIVE_DOCUMENT_SCHEMA_ID, StoredObjectiveDocument
from feedbax.contracts.project import BasedDocument, OrderDocumentMembers, Project, RootDocument, json_pointer_parts
from feedbax.contracts.project_resolution import compose_document
from feedbax.contracts.project_sources import pointer
from feedbax.contracts.set_document import SET_DOCUMENT_SCHEMA_ID, SetDocument
from feedbax.contracts.store_check import UnrolledFamilyAdvice
from feedbax.contracts.task_document import STORED_TASK_DOCUMENT_SCHEMA_ID, StoredTaskDocument


_COLLECTIONS = {
    STORED_OBJECTIVE_DOCUMENT_SCHEMA_ID: (StoredObjectiveDocument, "terms", "Objective terms"),
    SET_DOCUMENT_SCHEMA_ID: (SetDocument, "entries", "Set entries"),
    STORED_TASK_DOCUMENT_SCHEMA_ID: (StoredTaskDocument, "validation_trials", "validation trials"),
}


class _DraftMember(ValueError):
    pass


def _shape(value: object) -> Hashable:
    if isinstance(value, UnresolvedValue):
        raise _DraftMember()
    if isinstance(value, LiteralValue):
        return LiteralValue, type(value.value), value.unit
    if isinstance(value, BaseModel):
        return type(value), tuple((name, _shape(getattr(value, name))) for name in type(value).model_fields)
    if isinstance(value, dict):
        return dict, tuple((key, _shape(child)) for key, child in value.items())
    if isinstance(value, (tuple, list)):
        return type(value), tuple(_shape(child) for child in value)
    assert isinstance(value, Hashable)
    return type(value), value


def _groups(members: Mapping[str, StoredFamilyMember]) -> tuple[tuple[str, ...], ...]:
    groups: dict[Hashable, list[str]] = {}
    for key, member in members.items():
        if member.over is not None:
            continue
        try:
            shape = tuple(
                (name, _shape(getattr(member, name)))
                for name in type(member).model_fields if name not in {"over", "combination", "label"}
            )
        except _DraftMember:
            continue
        groups.setdefault(shape, []).append(key)
    return tuple(tuple(keys) for keys in groups.values() if len(keys) > 1)


def _local_writes(document: BasedDocument, collection: str) -> dict[str | None, str]:
    """Locate a difference that wrote the collection or one of its members."""
    paths: dict[str | None, str] = {}
    for path, change in document.differences.items():
        parts = json_pointer_parts(path)
        if parts[:1] == (collection,) and not isinstance(change, OrderDocumentMembers):
            member = parts[1] if len(parts) > 1 else None
            paths.setdefault(member, pointer(("differences", path)))
    return paths


def unrolled_family_advice(project: Project) -> tuple[UnrolledFamilyAdvice, ...]:
    advice = []
    for name, document in project.documents.items():
        if (entry := _COLLECTIONS.get(document.schema_id)) is None:
            continue
        schema, collection, plural = entry
        if isinstance(document, BasedDocument):
            local = _local_writes(document, collection)
            if not local:
                continue
            composed = compose_document(project.documents.get, name)
        else:
            local = {None: pointer(("payload", collection))}
            composed = document
        if not isinstance(composed, RootDocument):
            continue
        try:
            members = getattr(schema.model_validate(composed.payload), collection)
        except ValidationError:
            continue
        for group in _groups(members):
            path = local.get(None) or next((local[key] for key in group if key in local), None)
            if path is None:
                continue
            count = len(group)
            advice.append(UnrolledFamilyAdvice(
                code="unrolled_member_family", address=f"documents/{name}.json", path=path, members=group, count=count,
                detail=f"{count} {plural} share an expression and structure. Consider one family for {', '.join(group)}.",
            ))
    return tuple(advice)
